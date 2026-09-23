# Qwen3.8-Flash-Next 结构分析 + AutoRound MXFP8 / MXFP4 量化方案

- 模型：`/models/lkk/cri_models/Qwen3.8-Flash-Next`（bf16，131 分片，磁盘 336 GB，权重 360 GB）
- 架构：`Qwen4ExpForConditionalGeneration` / `model_type: qwen4_exp`（**模型名会骗人，config 不会**：它是 qwen4_exp，不是 qwen3_next）
- 参考产物：https://huggingface.co/INCModel3/Qwen3.8-Flash-Next-MXFP4-Mixed-CT-AutoRound （下文称 **HF 参考产物**，已逐项反解并复现）
- 先例 recipe：`/models/auto-quant/recipes/quant/Qwen__Qwen3.8-Flash-Next.json`（W4A16 线，2026-08-28 验证）
- 工具链与 `Qwen3.8-27B` 同一套：auto-round 0.16.0 @ env `muse_verify`；分析方法复用 `audit_precision_plan.py`（本轮已扩展支持 fused MoE）
- 本文所有数字均为**实测**（只读 `config.json` + safetensors header + HF 产物 config，复现命令见附录 A）
- **验证状态**：结构与精度分配审计=已完成；HF 参考产物反解=已完成（可精确复现）；**本模型的量化实跑与 vLLM 冒烟=尚未执行**（命令与守卫已就位，`DRY_RUN=1` 已验证）；§6 的 vLLM 结论来自本机源码静态阅读 + registry 探针 + 同机历史日志，非本模型端到端实跑

---

## 0. 结论速览

| 方案 | 组合 | 精度分配 | 产物体积 | 定位 |
|---|---|---|---|---|
| **M0 MXFP4-Mixed** | `--scheme MXFP8 --layer_config {mlp.experts:4bit}` | routed experts=MXFP4(W4A4)，attention/shared_expert/indexer=MXFP8，其余 BF16 | **≈180 GB** | **主推**；与 HF 参考产物逐项一致（73 728 个 4-bit 层，完全对上） |
| M1 MXFP8 | `--scheme MXFP8` | 量化区全 MXFP8 | ≈240 GB | ✅ **已端到端跑通**（TP=2 + `moe_backend=marlin`，§6.6/§6.7 有 bug 说明与自动选择规则）；最小并行度 TP=2 |
| M2 MXFP4 | `--scheme MXFP4` | 量化区全 MXFP4 | ≈178 GB | **不推荐**：只比 M0 省 1.5 GB，却把 attention 也压到 4 bit |

三条硬结论：

1. **压缩比被 PLE 的 n-gram 表封顶在 ~2×**。那张表 51.2 B 参数 / 102.4 GB bf16，是 embedding 查表不是 Linear，MX 系量化对它无从下手（`embed` 命中 auto-round 预定义 skip）。总参数 180 B 里，routed experts 120.8 B(67.1%) + n-gram 表 51.2 B(28.4%) = **95.6%**；剩下真正可 MX 量化的只有 ~4 B。所以 360 GB → 180 GB 已经是上限，别指望 MXFP4 把它变成 90 GB。
2. **`mlp.gate` 在这里必须 ignore，而且安全**。这是 MoE：`mlp.gate` 是 router（`[512,2560]`），子串不会命中 `mlp.experts.N.gate_proj`（那是 `mlp.experts.` 前缀）。与 27B（dense）的结论正好相反——**dense 上 `mlp.gate` 会误伤 `gate_proj`，MoE 上它是必保项**。
3. **`in_proj_a/in_proj_b` 依旧必须保 BF16**（N=48，vLLM 融合成 `in_proj_ba` N=96，`mm_mxfp8` 断言 N≥128）；`block_inject_weight`（N=4）同理。这两条在 27B 上是实测崩溃换来的，本模型结构同源，直接继承。

---

## 1. 模型结构

### 1.1 config 特征（day-0 判定项）

| 项 | 值 |
|---|---|
| `architectures` / `model_type` | `Qwen4ExpForConditionalGeneration` / `qwen4_exp`（VL 多模态） |
| 语言塔 | **MoE**：`num_experts=512`、`num_experts_per_tok=10`、`moe_intermediate_size=640`、`shared_expert_intermediate_size=640` |
| 层结构 | 48 层，`full_attention_interval=4` → **36× linear_attention(GDN) + 12× full_attention** |
| hidden / heads | 2560 / 24Q·2KV，`head_dim=256`，`partial_rotary_factor=0.25`，QK-norm |
| **hyper-connection** | `hc_count=4`、`hc_lowrank=320`；每层 `attn_hyper_connection` + `mlp_hyper_connection`（`input_mix_weight_down [320,10240]`、`input_mix_weight_up [10240,320]`、`block_inject_weight [4,10240]`）+ 顶层 `hyper_connection_mixer` |
| **PLE + n-gram** | `ple_layer_ids=[2]`（只在第 2 层）、`ple_embed_dim=2560`、`ngram_size=3`、`ngram_vocab_size_base=20 000 000`、`split_ngram_parts=128`、`heads_per_ngram=8` |
| **稀疏注意力 indexer** | `indexer_n_heads=4`、`indexer_kv_heads=1`、`indexer_head_dim=128`、`indexer_budget=2048`、`indexer_compress_ratio=4`；仅 full_attention 层带 `self_attn.indexer.*` |
| MTP | `mtp.hybrid=true`、1 层 full_attention，**自带 512 experts** |
| 视觉塔 | 与 27B 同型（27 层，hidden 1152），`out_hidden_size=2560` |
| 词表 / 上下文 | 248320 / 262144；embed 与 lm_head 不共享 |

### 1.2 张量清单（实测 shape，`n`=实例数）

| 张量模式 | n | shape | 参数 |
|---|---|---|---|
| `layers.N.mlp.experts.gate_up_proj` | 48 | **3-D** `[512, 1280, 2560]` | 80 530.6 M |
| `layers.N.mlp.experts.down_proj` | 48 | **3-D** `[512, 2560, 640]` | 40 265.3 M |
| `layers.N.ple.ple_embedding.ngram_embedding.shard_{0..127}` | 128 | `[2500012, 160]` | 51 200.2 M |
| `layers.N.linear_attn.in_proj_qkv` | 36 | `[10240, 2560]` | 943.7 M |
| `embed_tokens` / `lm_head` | 1 / 1 | `[248320, 2560]` | 635.7 M 各 |
| `layers.N.linear_attn.in_proj_z` / `out_proj` | 36 / 36 | `[6144,2560]` / `[2560,6144]` | 566.2 M 各 |
| `layers.N.self_attn.q_proj` | 12 | `[12288, 2560]`（含 output gate） | 377.5 M |
| `layers.N.self_attn.o_proj` | 12 | `[2560, 6144]` | 188.7 M |
| `*hyper_connection.input_mix_weight_{down,up}` | 96 | `[320,10240]` / `[10240,320]` | 157.3 M ×4 |
| `layers.N.mlp.shared_expert.{gate,up,down}_proj` | 144 | `[640,2560]` / `[2560,640]` | 78.6 M 各 |
| `layers.N.mlp.gate`（router） | 48 | `[512, 2560]` | 62.9 M |
| `layers.N.self_attn.indexer.index_qk_proj` | 12 | `[640, 2560]` | 19.7 M |
| `layers.N.linear_attn.in_proj_{a,b}` | 36 + 36 | **`[48, 2560]`** | 4.4 M 各 |
| `*hyper_connection.block_inject_weight` | 48 | **`[4, 10240]`** | 2.0 M |
| `mtp.layers.0.mlp.experts.{gate_up,down}_proj` | 2 | `[512,1280,2560]` / `[512,2560,640]` | 2 516.6 M |
| `mtp.fc_embedding` / `mtp.fc_hidden` | 2 | `[2560, 2560]` | 6.6 M 各 |
| 1-D（`hc_norm`、`*_layernorm`、`A_log`、`dt_bias`、`ngram_heads_*`） | — | — | 极小 |
| 3-D / 5-D conv（`linear_attn.conv1d`、`ple.conv1d`、`visual.patch_embed.proj`） | — | — | 不参与 MX |

> **fused experts**：权重以 `[E, N, K]` 3-D 形式存储。auto-round 的 `split_fused_expert_tensors` 会先拆成
> `experts.<i>.{gate,up,down}_proj` 再量化（`gate_up_proj` 沿 N 劈成 gate/up 两半），
> 所以导出产物的 `config_groups.targets` 是 73 728 个逐专家层名，而不是 96 个 fused 张量。

### 1.3 参数量与占比（总 **180.000 B**，bf16 360.0 GB）

| 模块 | 参数 | 占比 |
|---|---|---|
| Routed experts（48 层 × 512） | 120.796 B | **67.11 %** |
| PLE n-gram embedding 表（128 shards） | 51.200 B | **28.44 %** |
| MTP routed experts（512） | 2.517 B | 1.40 % |
| GDN linear_attn（36 层） | 2.087 B | 1.16 % |
| hyper-connection（48×2 + 顶层） | 0.660 B | 0.37 % |
| `embed_tokens` | 0.636 B | 0.35 % |
| `lm_head` | 0.636 B | 0.35 % |
| full self_attn（含 indexer，12 层） | 0.617 B | 0.34 % |
| Vision tower | 0.449 B | 0.25 % |
| `shared_expert`（+ gate） | 0.241 B | 0.13 % |
| MTP 其余 | 0.066 B | 0.04 % |
| MoE router（`mlp.gate`） | 0.063 B | 0.03 % |
| PLE dense 投影 | 0.033 B | 0.02 % |

- **可 MX 量化的 Linear 只有 ~126.7 B（70.4%）**，其中 95.3% 是 routed experts → 量化的本质就是压 experts 那一块。
- 每 token 活跃参数 ≈ **6.7 B**（dense 部分 + 10/512 experts，不含 embedding 查表）→ 180B-A7B 档，推理是内存带宽受限、算力不太受限，正是 MXFP4 想要的负载形状。
- n-gram 表虽只被"查表"，但 102 GB 必须常驻显存 → **这是部署显存的第一决定因素，不是权重精度**。

---

## 2. HF 参考产物反解（INCModel3/Qwen3.8-Flash-Next-MXFP4-Mixed-CT-AutoRound）

从 `quantization_config.json` 实测读到的配置：

| 项 | 值 |
|---|---|
| `format` / `quant_method` | `mixed-precision` / `compressed-tensors` |
| `group_0` | **W4A4**，`group_size=32`，`scale_dtype=torch.uint8`(E8M0)，`dynamic` 激活，targets = **73 728** 个 `layers.*.mlp.experts.*.{gate,up,down}_proj`（48×512×3） |
| `group_1` | **W8A8**，g32，targets = `["Linear"]`（其余全部） |
| `ignore` | **2 295** 项：`lm_head`、`embed_tokens`、visual 全部、`mtp.*` 全部（含 1536 个 mtp experts）、`ple...ngram_embedding.shard_*` 128、`*_hyper_connection.{block_inject_weight,input_mix_weight_down,input_mix_weight_up}` 144、`mlp.gate` 48、`mlp.shared_expert_gate` 48、`linear_attn.{conv1d,in_proj_a,in_proj_b}` 108 |
| `kv_cache_scheme` | `null`（未做 KV 量化） |

其 README 给出的生成命令（本方案 M0 就是它的复现）：

```bash
auto_round --model_name Qwen/Qwen3.8-Flash-Next --model_free --scheme MXFP8 \
  --ignore_layers "visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection" \
  --layer_config "{mlp.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor --output_dir ./Qwen3.8-Flash-Next-MXFP4
```

**复现校验（本文实测）**：用同一 ignore + layer_config 跑 `audit_precision_plan.py`，预测

```
bits=4   120.796 B (67.1%)  层数 73728     ← 与 group_0 targets 数完全一致
bits=8     2.929 B ( 1.6%)  层数 312       ← attention + shared_expert + indexer
bits=16   56.270 B (31.3%)  层数 2257      ← 与 ignore 集合逐项对齐（差异仅为专家下标展开方式）
```

也就是说：**HF 参考产物的精度分配已被完全解释**——pattern 集合逐项对得上，4-bit experts 的 73 728 层可精确复现。
它没有对 attention 做任何额外保护（都落在 MXFP8），MTP/PLE/hyper-connection 整块保明文。
（**但**本机 auto-round 0.16 实际落盘时会额外把 `mlp.shared_expert.*` 144 层排除，见 §6.6 附注 →
本机产出的 M0 与参考产物不是字节同构，8-bit 覆盖面略窄。）

注意 `--layer_config` 只写 `bits/data_type` 时，llm_compressor 会按有效 weight bits 重建整组 preset → 得到的是 **W4A4**（激活也 4 bit），不是 W4A8。参考产物正是 W4A4，所以别"好心"去补 `act_bits:8`。

---

## 3. auto-round 对本模型的自动处理

- **fused 3-D experts 自动拆分**（见 §1.2 注），拆出来的 per-expert 层共享同一 scheme。
- **预定义 skip** `_BLOCK_NAME_TO_IGNORE = ("shared_expert_gate.", ".gate.", "embed", "conv")`：
  `mlp.gate.weight`、`mlp.shared_expert_gate.weight`、`embed_tokens`、`ngram_embedding`（含 `embed`）、`conv1d`（含 `conv`）都会**自动**保明文——所以 §4 清单里这几项写出来是"显式表达意图 + 与参考产物对齐"，不写也会被跳过；但 `in_proj_a/b`、`block_inject_weight`、`hyper_connection`、`mtp`、`visual` **必须手写**。
- VLM 默认只量化文本塔、lm_head 默认不量化。
- **`--model_free` 是本模型的现实路径**：180 B / 360 GB 走调优路径要真加载 + 校准前向，先例（W4A16 tuning）实测 10 小时级且**精度没有比 RTN 更好**（见 pinned recipe：mean 0.8800 vs RTN 0.8819）→ 除非 M0/M1 掉点超阈值，不要碰调优。

---

## 4. ignore 清单逐项依据

| 模式 | 实例 | 为什么保 BF16 |
|---|---|---|
| `visual` | 112 | 文本评测不走，mm 路径无量化 kernel |
| `lm_head` / `embed_tokens` | 1 / 1 | 输出层与输入表常规保高精度（各 0.636 B） |
| `mlp.gate` | 48 | **router**：512 选 10 的 softmax 打分，量化会直接改路由分布 |
| `mlp.shared_expert_gate` | 48 | `[1,2560]` 门控，N=1，任何 MX kernel 都吃不下 |
| `in_proj_a` / `in_proj_b` | 36 / 36 | N=48 → vLLM 融合 `in_proj_ba` N=96 < 128，`mm_mxfp8` 断言崩（27B 实测） |
| `block_inject_weight` | 48 | `[4,10240]`，N=4，同理不可量化 |
| `hyper_connection` | 96 + 顶层 | 结构性模块（残差混合），且引擎侧融合；`input_mix_weight_*` 虽是合法 Linear，先例与参考产物都保明文，收益 <1.3 GB |
| `ple` | 128 + 2 | n-gram 表是 embedding 查表（102 GB，MX 不适用）+ PLE 投影；整块保明文 |
| `mtp` | 全部 | 投机解码模块，自带 512 experts（2.5 B），主路径不受益；开了 spec decode 才需要它，届时另做验证 |

**别写 `mlp.gate$` 之外的锚定形式**：本模型 `mlp.gate` 子串是安全的（已实测匹配集合 = 48 个 router），但如果以后复用到 dense 变体会立刻误伤。

---

## 5. 量化命令

### 5.1 M0 · MXFP4-Mixed（主推，复现 HF 参考产物）

```bash
export PATH=/models/miniforge3/envs/muse_verify/bin:$PATH
export HF_HOME=/models/huggingface
export CUDA_VISIBLE_DEVICES=2                 # 先 nvidia-smi 确认空闲，别抢卡
export AR_MODEL_FREE_SHARD_PARALLELISM=1      # 360GB 流式重写，限并发控 host RAM

auto-round \
  --model_name /models/lkk/cri_models/Qwen3.8-Flash-Next \
  --model_free \
  --scheme MXFP8 \
  --layer_config '{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}' \
  --ignore_layers visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection \
  --format llm_compressor \
  --device_map auto \
  --output_dir /models/lkk/cri_models/Qwen3.8-Flash-Next-MXFP4-Mixed-AutoRound
```

### 5.2 M1 · 纯 MXFP8（保守）

同 5.1，去掉 `--layer_config`，`--output_dir .../Qwen3.8-Flash-Next-MXFP8-AutoRound`。

### 5.3 M2 · 纯 MXFP4（不推荐）

同 5.1，`--scheme MXFP4` 且去掉 `--layer_config`。**只比 M0 省 1.5 GB**（量化区里非 experts 的只有 2.9 B），却把 attention/indexer/shared_expert 全压到 4 bit —— 精度风险显著上升，收益可忽略。

### 5.4 一键脚本（含前置审计与导出核对）

```bash
cd /models/lkk/cri_models
DRY_RUN=1 ./quant_Qwen3.8-Flash-Next-mxfp.sh mxfp4_mixed   # 只看审计表 + 命令
./quant_Qwen3.8-Flash-Next-mxfp.sh mxfp4_mixed             # M0
./quant_Qwen3.8-Flash-Next-mxfp.sh mxfp8                   # M1
```

---

## 6. vLLM 支持判定（评测能不能跑）

### 6.1 本机环境：**已经支持，不用新建环境**

只读探针（不加载权重、不起引擎）：

```bash
CUDA_VISIBLE_DEVICES="" /models/miniforge3/envs/muse_verify/bin/python -c "
from vllm.model_executor.models.registry import ModelRegistry as R
a=R.get_supported_archs()
print({k:(k in a) for k in ['Qwen4ExpForConditionalGeneration','Qwen4ExpForCausalLM','Qwen4ExpMTP']})
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS as Q
print(sorted(m for m in Q if 'mxfp' in m or m in ('compressed-tensors','inc','fp8')))"
# {'Qwen4ExpForConditionalGeneration': True, 'Qwen4ExpForCausalLM': True, 'Qwen4ExpMTP': True}
# ['compressed-tensors', 'fp8', 'gpt_oss_mxfp4', 'inc', 'modelopt_mxfp8', 'mxfp4', 'mxfp8']
```

本机 vLLM = `/models/lkk/muse_support/vllm`（`0.28.1rc1.dev312+g41848caa6.d20260917`，vendor 树，
虽挂着 0.28.1 的号但**已含 qwen4_exp**，registry 在 `registry.py:114/580/670` → `vllm.models.qwen4_exp`）。
配套：torch 2.13.0+cu130、triton 3.7.1、flashinfer 0.6.18、transformers 5.16.1、4×B300(sm103)。

**结论：MXFP8 / MXFP4-Mixed 产物可以直接在本机 `muse_verify` 环境评测，无需重建环境。**
§7 的重建流程是给"要脱离这棵 vendor 树 / 换机器 / 要干净可复现环境"的情况用的。

### 6.2 上游状态（2026-09-21 核实，**旧记录作废**）

| 项 | 结论 |
|---|---|
| 支持 PR | `vllm-project/vllm#53896` **[Model] Support Qwen3.8-Flash-Next** → **已合并**，`merged_at=2026-08-31T05:57:56Z`，merge commit `e126687a` |
| 首个包含的 tag | `v0.29.0rc1`；**首个 stable = `v0.29.0`**（v0.28.x 全部没有） |
| PyPI | `vllm==0.29.0`（2026-09-09 上传，当前最新）→ **纯 pip 可用，不需要源码编译**；PyPI 尚无 0.30.0 |
| 上游文件形态 | 不是单文件 `models/qwen4_exp.py`，而是包 `vllm/models/qwen4_exp/`（`config.py`、`common/`、`nvidia/`、`amd/`），与本机树一致 |
| Docker | `vllm/vllm-openai:v0.29.0` 已含支持（与 `latest` 同 digest）；**8/26 的 `:qwen38-flash-next` 专用 tag 是 PR 分支临时构建，比 merge 还早 5 天，别再用了**；要 9/14–9/17 那批 Qwen4Exp 改动（#55309/#55557/#57273）只能上 `cu129-nightly` 或等 v0.30.0 |
| GDN kernel | `fused_gdn_decode_kernel.cu` **不是本 PR 新增**（v0.28.0 已有，PR 只改了 `+26 −10`）；`VLLM_GDN_DECODE_KERNEL` 默认 `cuda` 但会**自动回落 triton**，只有你显式设成 `cuda` 才会在不满足条件时抛错 → **别显式设它** |

### 6.3 量化通路（为什么 MXFP4-Mixed 这条路是"已知配置"而不是"理论上能过"）

- qwen4_exp 模型侧**没有自带 quant method**，只把 `vllm_config.quant_config` 透传给标准 Linear/`FusedMoE`
  （`vllm/models/qwen4_exp/nvidia/model.py:160-171`，MoE 类 `Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock)`）
  → 所以 v0.29.0 的 `mxfp4` / `mxfp8` / `compressed-tensors` / `inc` 全在可用集合里
  （`vllm/model_executor/layers/quantization/__init__.py:13-46`）。
- **MoE 的 MXFP4 有专用后端与优先级**：`FLASHINFER_TRTLLM_MXFP4_MXFP8 → DEEPGEMM_MXFP4 → MARLIN/BATCHED_MARLIN`
  （`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:344-366`）；compressed-tensors 侧有
  `compressed_tensors_moe_w4a4_mxfp4.py` / `_w8a8_mxfp8.py`，正好对上 M0 的两个 config_group。
- 上游甚至**为本模型形状写死了特例**：`oracle/mxfp4.py:477-507` 的 `_requires_qwen38_tep8_emulation`
  里 `E=512, H=2560, N=640` W4A4 —— 就是 Flash-Next 的 experts 形状。
- `triton_kernels` **不是硬依赖**：只有 MXFP4 MoE 选中 TRITON 后端才 import（`oracle/mxfp4.py:48` 有
  `has_triton_kernels()` 保护）。本机日志里那行 `No module named 'triton_kernels.matmul_ogs'` 属无害告警。

### 6.4 本机 vLLM 代码级约束（这就是 ignore 清单长这样的原因）

| 约束 | 代码位置 | 对本方案的含义 |
|---|---|---|
| **vLLM 的 `ignore` 只认「完整层名」或 `re:` 前缀，裸子串静默失效** | `compressed_tensors/utils/config_utils.py:300-322`（`use_fnmatch=False`、`check_contains` 未传）+ `compressed_tensors/utils.py:53-70` | 命令里写 `mlp.gate` 是**给 auto-round 的 pattern**；能不能生效取决于导出产物里有没有展开成完整层名。HF 参考产物 2295 条全是完整层名 ✓ → 所以**每次量化后必须跑 `check_quant_export.py`**，这一步不是形式主义 |
| router / shared_expert_gate / hyper-connection / PLE 在 vLLM 侧**硬编码 `quant_config=None`** | `qwen3_next.py:170-184`（gate、shared_expert_gate）、`qwen4_exp/nvidia/hyperconnection.py:102,113,122`、`ple_layer.py:130-153`（CT 下直接返回 None） | 这些模块**必须明文**。若产物里存了它们的 `weight_scale`，加载期报 `There is no module or parameter named ...`（白名单只收 `.q/.k/.v_scale` 和 `_weight_scale`，**不匹配 `.weight_scale`**）。→ 本文 ignore 里的 `mlp.gate,mlp.shared_expert_gate,hyper_connection,block_inject_weight,ple` 是**硬约束不是保守** |
| `shared_expert` 与 routed experts 同一条 scheme | `qwen3_next.py:203-213` 把 quant_config 透传进去 | M0 把 shared_expert 落在 group_1(MXFP8) 是正常且必要的，别照抄 W4A16 先例把它保 BF16 |
| 融合层的分片 scheme 必须一致 | `compressed_tensors/utils.py:65-69` `Found different quantization schemes for the shards`；`packed_modules_mapping`（`nvidia/model.py:619` 含 `in_proj_ba`） | `in_proj_a` 与 `in_proj_b` 必须**同时** ignore；`gate_up_proj` 拆出的 gate/up 也必须同组 |
| **`in_proj_ba` N=96，且 TP>1 更小** | `qwen_gdn_linear_attn.py:576-598`（`output_sizes=[num_v_heads]*2` → N=96）、`:600-616` `maybe_disable_tp` 白名单**不含 compressed-tensors** → TP=4 时每卡 N=24 | MXFP8 的 `mxfp8/flashinfer.py:150-160`(Cutedsl) 与 `:66-76`(Cutlass) 都有**运行期** `assert 128 <= N`，且 `can_implement:117-119` 恒真、config 不带形状 → 选型阶段不规避，只在首次 forward / CUDA-graph 捕获时爆。**只能靠量化侧 ignore 解决** |
| **不要开 expert parallel** | `experts/cutlass_moe.py:1041-1044` 要求 `ep_size==1`，而 `compressed_tensors_moe_w4a4_mxfp4.py:54` 只看设备不看并行度 | EP>1 没有干净回退，直接崩 |
| CT 的 MXFP4 Linear **无条件按 W4A4 执行** | `schemes/compressed_tensors_w4a4_mxfp4.py:41-43`（固定 `kMxfp4Dynamic`） | 想要"MXFP4 权重 + BF16 激活"在 sm103 的 CT 路径上**拿不到**，且不报错（只有 Marlin 回退会 warning）→ 别指望 `act_bits:16` 能救精度 |
| MTP draft 的 ignore 重映射不覆盖 CT | `nvidia/mtp.py:63-78, 125-138` 只处理 `ignored_layers`/`exclude_modules`，而 `CompressedTensorsConfig` 用的是 `.ignore` | 评测阶段**不要开 spec decode**（`--speculative-config`）；产物也别量化 mtp（已 ignore） |
| PLE ngram 表按 vocab 维 TP 切分、形状严格校验 | `ple_layer.py:314-323, 482-512`（`shard_{i}.weight`，`i < 128`） | 102 GB 表**关不掉也不许量化**（`envs.py` 无任何 ple/ngram 开关；`--hf-overrides` 删 `ple_layer_ids` 会让 ckpt 张量变孤儿而崩）。TP=4 约 25.6 GB/卡，`gpu_memory_utilization` 要留够 |

启动期硬红线（不满足直接抛）：**PP 必须=1**（PLE 要原始 input_ids，`models/config.py:879-884`）、**dtype 必须 bfloat16**、**KV cache 必须 BF16（拒绝任何 KV 量化**，`nvidia/qsa.py:190-193`——所以 27B 那套 `-FP8KV`/`-FP8Attn` 变体在本模型上**一定起不来**）、不支持 PCP/DCP、不支持 SP-MoE、`tp_size ≤ 512` 且 TP 需整除 `linear_num_value_heads=48` 与 24 heads、`--mamba-cache-mode` 不能是 `all`。

兜底（产物已成型又不想重量化时）：`VLLM_DISABLED_KERNELS=FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel,MarlinMxfp8LinearKernel` → 退 `EmulationMxfp8LinearKernel`（加载期反量化成 BF16，慢但对）；**全局生效**，只用于定位问题。

**同机实证**：`/models/lkk/server_Qwen3.5-397B-A17B_mxfp4_fp8kv.log`（8-27，GDN+MoE 同族、同为 CT MXFP4-Mixed 产物）里明确打出
`Using FlashInferCutedslMxfp8LinearKernel for MXFP8 GEMM` + `Using CutlassExpertsMxfp4 for MXFP4 MoE` + `Application startup complete`（TP=4）
→ M0 的两个 config_group 在本机这棵树上**都有实跑记录**，不是纸面推断。

### 6.5 评测参数

上游自带的评测配置可作权威参照：`tests/evals/qwen4_exp/configs/Qwen3.8-Flash-Next-FP8.yaml`
→ `--tensor-parallel-size 4` + `--trust-remote-code`，生成参数 `temperature=0.7, top_p=0.8, top_k=20,
presence_penalty=1.5, max_tokens=32768, chat_template_kwargs.enable_thinking=true`。
**注意上游是开 thinking 评测的**；本文与 27B 保持同口径默认 `enable_thinking=False`（要切上游口径：`ENABLE_THINKING=1`）。

```bash
cd /models/lkk/cri_models
SMOKE_ONLY=1 ./eval_Qwen3.8-Flash-Next-quant.sh Qwen3.8-Flash-Next-MXFP4-Mixed-AutoRound
TP=2 CUDA_VISIBLE_DEVICES=1,2 TASKS=piqa LIMIT=16 ./eval_Qwen3.8-Flash-Next-quant.sh <产物>
./eval_Qwen3.8-Flash-Next-quant.sh <产物>                                  # 正式跑：默认不冒烟；跑 gsm8k + TASKS
RUN_GSM8K=0 ./eval_Qwen3.8-Flash-Next-quant.sh <产物>                      # 只跑 TASKS（省一次完整加载）
TP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ./eval_Qwen3.8-Flash-Next-quant.sh <产物>       # 全量（对齐上游 TP=4）
ENVBIN=/models/lkk/venv_qwen38fn/bin ./eval_Qwen3.8-Flash-Next-quant.sh <产物>    # 用 §7 的干净环境
```

固定口径：`language_model_only=True`、`dtype=bfloat16`、`reasoning_parser=qwen3`、`trust_remote_code=True`、
`seed=42`、gsm8k 走 `--apply_chat_template --fewshot_as_multiturn`；**禁** KV 量化 / spec decode / expert parallel / PP / SP。
`--model_args` 以**整体 JSON** 传入（这版 lm_eval 对含 `{` 的逗号 k=v 串会直接 `Invalid JSON` 报错，见 §8）。
显存：M0 ≈180 GB → TP2 起步（含 102 GB PLE 表）；bf16 baseline 360 GB → TP4。

### 6.6 【实跑踩坑】纯 MXFP8 产物 + TP>1 崩在 vLLM 的 MoE scale swizzle

2026-09-21 09:46 实测（`eval_Qwen3.8-Flash-Next-MXFP8-AutoRound.log`，TP=2）：

```
compressed_tensors_moe_w8a8_mxfp8.py → convert_to_fp8_moe_kernel_format
  → prepare_fp8_moe_layer_for_fi → _shuffle_mxfp8_moe_weights   (flashinfer_utils.py:459-460)
w2_scale_out = torch.stack(w2_scale_shuffled).reshape(w2_scale.shape)
RuntimeError: shape '[512, 2560, 10]' is invalid for input of size 15728640
```

### 因果链（每一环都有依据）

1. 产物侧合规：`mlp.experts.<i>.down_proj.weight` = `F8_E4M3 (2560, 640)`，`weight_scale` = `U8 (2560, 20)`（20 = 640/32），`check_quant_export.py` 通过。**不是量化错误。**
2. TP=2 → `down_proj` 按输入维切分 → 每卡 `w2` = `[2560, 320]`，`w2_scale` = `[2560, 10]`（**10 = 640/TP/32**）。
3. MXFP8 MoE 走 FlashInfer TRTLLM 后端（`oracle/mxfp8.py:19-31` 优先级第一），预处理里对 scale 调
   `shuffle_matrix_sf_a(..., epilogue_tile_m=128)` 做 TRT-LLM 的 **128×4 swizzle，列数向上补到 4 的倍数**。
4. 紧接着却 `.reshape(w2_scale.shape)`，即按**补前**的 `[512, 2560, 10]` 还原 → 元素数不匹配 → 崩。
5. CPU 实测 `shuffle_matrix_sf_a` 的 padding 行为（`flashinfer 0.6.18`）：

   | 输入 | numel | 输出 numel | 是否被 pad |
   |---|---|---|---|
   | `(2560, 20)` | 51 200 | 51 200 | 否 → TP=1 安全 |
   | `(2560, 10)` | 25 600 | **30 720** | 是（10→12）→ TP=2 崩，正合日志里的 512×30720=15 728 640 |
   | `(2560, 5)` | 12 800 | **20 480** | 是（5→8）→ TP=4 崩 |

**通用触发条件**：每卡 scale 列数 `moe_intermediate_size/(TP·32)` 不是 4 的倍数。本模型 `moe_intermediate=640`
→ TP=1 得 20 ✓，**TP=2 得 10 ✗，TP=4 得 5 ✗** ⇒ 用**默认的 FlashInfer TRTLLM 后端**时，这份纯 MXFP8 产物在 TP>1 下必崩。

**TP=1 实测两次：不可行，但原因不是权重**（这段结论我错过两次，把过程记全）：

| 尝试 | 结果 |
|---|---|
| TP=1，`GPU_MEM=0.95`，`max_num_batched_tokens=16384` | `Model loading took 219.94 GiB / 224 s` —— **权重确实装得下**（0.85 才真装不下：预算 228.3 GiB vs 权重 223.9 GiB 只剩 4.4 GiB），且选中原生 `FLASHINFER_TRTLLM`、**没触发 swizzle bug**；随后 OOM：`Failed to run autotuning code block: CUDA out of memory. Tried to allocate 95.37 GiB` |
| TP=1，`GPU_MEM=0.95`，`max_num_batched_tokens=2048` | **申请量一模一样 95.37 GiB** → 我"autotune 峰值随 token 数放大"的假说被自己的实验证伪 |

那 95.37 GiB 是什么：**PLE 的 n-gram 表** = 51 200.25M × 2 B = 102.40 GB = **95.37 GiB**，精确吻合。
TP=1 时它无法切分，首帧/编译阶段需要**整份额外拷贝** → 219.94 + 95.37 ≫ 268.6 GiB；
TP≥2 时该表按 vocab 维切分（`PLEVocabParallelEmbedding`，128 shard），峰值减半到 47.68 GiB/卡 → 够用。
⇒ **M1 在本机的现实最小并行度是 TP=2**。`eval_Qwen3.8-Flash-Next-quant.sh` 已把这条做成守卫
（产物里有 `ngram_embedding` 且 `TP==1` → 直接拒绝，除非 `FORCE_TP1=1`）。

### 为什么 27B 的 MXFP8 没事、M0 预计也没事

- 27B 是 dense，MXFP8 只走 **Linear** 核（`FlashInferCutedslMxfp8LinearKernel`），那条路径自己做了
  `padded_n` 处理，且没有 MoE 预处理。
- M0（MXFP4-Mixed）的 experts 走 `CompressedTensorsW4A4Mxfp4MoEMethod` → `CutlassExpertsMxfp4`，
  scale swizzle 是 `swizzle_mxfp4_scales(...).reshape(padded_N, -1)`——**显式按 pad 后的形状还原**，
  没有这个 bug。同机实证：`/models/lkk/server_Qwen3.5-397B-A17B_mxfp4_fp8kv.log:46,367`
  （GDN+MoE、CT MXFP4-Mixed、TP=4、`Using CutlassExpertsMxfp4 for MXFP4 MoE` → `Application startup complete`）。

### 绕行方案与实测状态

| 方案 | 做法 | 状态 |
|---|---|---|
| A | TP=1（每卡 20 列，能规避 swizzle bug） | ❌ **实测不可行**：不触发 bug 了，但 PLE n-gram 表要整份第二份拷贝 95.37 GiB → OOM（与 batch 无关，见上表） |
| **B（M1 的正解）** | **TP=2 + `moe_backend=marlin`**（脚本自动选） | ✅ **端到端跑通**：`Using 'MARLIN' MxFp8 MoE backend` → 权重 111.49 GiB/rank → `Available KV cache memory: 109.69 GiB` → warmup 8.94 s → CUDA graph PIECEWISE+FULL 100% → **`SMOKE OK`**，3 条生成语义正确（Paris/Berlin…；`4+4=`→`8. 8+8=16…`；可运行的 quicksort）。日志 `smoke_fn_mxfp8_marlin.log` |
| B' | 其它绕开后端 | `deep_gemm`、`humming_kernels` 本机**未安装**；`triton` 被 CUDA 拒绝（`No supported MXFP8 expert class for TRITON_MXFP8`）。脚本用 `importlib.find_spec` 现探现选，装了就会自动优先用 |
| C | **改跑 M0 MXFP4-Mixed**：experts 走 `CutlassExpertsMxfp4`（其 swizzle 显式按 `padded_N` 还原），压根不经过这个函数；产物 ~180 GB → TP=2 也很宽松 | 待跑（`./quant_Qwen3.8-Flash-Next-mxfp.sh mxfp4_mixed`） |
| D | 上游补丁：`_shuffle_mxfp8_moe_weights` 不该 `.reshape(未 pad 形状)`（TRT-LLM 要的就是 padded SF） | 属改 vendor 树 `/models/lkk/muse_support/vllm`，需单独决策；值得先查 upstream 是否已修 |

`eval_Qwen3.8-Flash-Next-quant.sh` 已内置**自动绕开**：启动前读产物的 `quantization_config` + 索引，判定为
"MXFP8 experts + TP>1" 时自动加 `kernel_config={"moe_backend":"marlin"}`（本机唯一可用的绕开后端）；
`MOE_BACKEND=none` 可强制走 vLLM 自动选型以复现该崩溃，`MOE_BACKEND=<x>` 手动指定。
lm_eval 侧这条路是通的：它的 `split_top_level` 只切"括号外的逗号"，值走 `ast.literal_eval`，
所以 `kernel_config={"moe_backend":"marlin"}` 能原样进 `LLM(**model_args)`（已单测验证）。

### 附：与预测分配的一处实测偏差（不影响本次崩溃）

本产物实测 **73 896** 个量化层，而审计预测 **74 040**，差 **144** = `mlp.shared_expert.{gate,up,down}_proj × 48`：
auto-round 的匹配器（`_PatternMatcher`）对这 144 层返回的是 `quant / bits=8`，但导出时它们被列进了 `ignore`
（产物里 147 个 `shared_expert.` 条目、0 个 `weight_scale`）→ 说明 model_free 的落盘阶段还有一层我没还原的
shared_expert 处理规则。影响很小（144×~1.6M 参数留在 BF16，体积 +0.12 GB），且对 vLLM 更保险；
但与 HF 参考产物（shared_expert 走 group_1 的 MXFP8）**不一致**，所以 §2 的"完全复现"要精确表述为：
**4-bit experts 部分逐层复现（73 728 一致）；8-bit 覆盖集合本机略窄（少 144 层 shared_expert）**。

### 6.7 eval 脚本的后端自动选择规则（已实现）

`eval_Qwen3.8-Flash-Next-quant.sh` 启动前读产物的 `config.json` + `quantization_config` + 分片索引，按下述
确定性规则决定 MoE 后端与 TP，**不靠猜也不靠手记**：

```
cols        = moe_intermediate_size / TP / 32        # 每卡 MXFP8 MoE 的 scale 列数
expert_bits = experts 所在 config_group 的 weights.num_bits
HAS_PLE     = 索引里是否存在 *ngram_embedding*

若 expert_bits == 8 且 HAS_PLE 之类前置成立：
    cols 为整数且 cols % 4 == 0 → MOE_BACKEND=none（自动选型 → 原生 FlashInfer TRTLLM，最快）
    否则                        → 按 deep_gemm > marlin > humming 顺序，用 importlib.find_spec
                                  探本机真装得上的那个（当前命中 marlin）并注入
                                  kernel_config={"moe_backend": "<b>"}
若 expert_bits == 4（MXFP4-Mixed）→ 不涉及该 bug → 自动选型
TP == 1 且 HAS_PLE → 直接拒绝（PLE 整份拷贝 95.37 GiB 必 OOM），FORCE_TP1=1 可强行试
```

要点：

- 判据是 **`cols % 4`**，不是"TP>1"这个粗略代理 —— 换 `moe_intermediate_size` 不同的模型结论会自动变对
  （例如 `moe_intermediate=2048` 时 TP=4 → cols=16 → 原生后端本就安全）。
- 绕开后端是**现探**的（`importlib.util.find_spec`），所以将来 `pip install deep_gemm` 之后无需改脚本，
  会自动优先选更快的 `deep_gemm`。
- 显式设 `MOE_BACKEND=<名字>` 永远优先于自动判定（想复现崩溃就 `MOE_BACKEND=flashinfer_trtllm`）；
  `MOE_BACKEND=none` 表示"交给 vLLM 自动选型"。
- 该 dict 能穿过 lm_eval：其 `split_top_level` 只切括号外的逗号，值走 `ast.literal_eval`（已单测）。

## 7. 干净环境构建（系统 python venv + uv）

脚本：`./setup_vllm_env_qwen38fn.sh`（`DRY_RUN=1` 可先看命令）。手工等价步骤：

```bash
# 0) 前置：系统 python 3.12.3（不用 conda）+ uv 0.12.13；/models 剩余 508 GB（venv 约 20-25 GB）
/usr/bin/python3 -c "import venv,ensurepip;print('venv OK')"
uv --version          # 没有就: curl -LsSf https://astral.sh/uv/install.sh | sh

# 1) 用系统 python 建 venv
/usr/bin/python3 -m venv /models/lkk/venv_qwen38fn
export VIRTUAL_ENV=/models/lkk/venv_qwen38fn        # 让 uv 装进这个 venv

# 2) 装 vLLM（B300=sm103 必须 cu130 构建 → 用 uv 的 --torch-backend）
uv pip install --torch-backend=cu130 "vllm==0.29.0"
#   需要 9 月中旬之后的 Qwen4Exp 改动时改用 nightly：
#   uv pip install --torch-backend=cu130 --pre --extra-index-url https://wheels.vllm.ai/nightly "vllm>=0.29.0"

# 3) flashinfer 的 cubin 不在 PyPI（SM100+ 的 MXFP4/MXFP8 快路径需要）
uv pip install --extra-index-url https://flashinfer.ai/whl/ "flashinfer-cubin==0.6.18"

# 4) 量化 + 评测侧
uv pip install "auto-round>=0.15.0" "lm-eval" "transformers>=5.16.1" accelerate

# 5) 自检（不加载权重）
$VIRTUAL_ENV/bin/python - <<'PY'
import torch, vllm
from vllm.model_executor.models.registry import ModelRegistry as R
print("vllm", vllm.__version__, "| torch", torch.__version__)
print("qwen4_exp:", [a for a in ("Qwen4ExpForConditionalGeneration","Qwen4ExpMTP")
                      if a in R.get_supported_archs()])
print("cc:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "no gpu")
PY
```

要点与坑：

- **`--torch-backend=cu130` 是必须的**：默认解析可能给到 cu12x 构建，sm103(B300) 上会缺 kernel。
  装完务必核对 `torch.version.cuda == 13.0`。
- vllm 0.29.0 会自带 `torch==2.13.0` / `transformers>=5.10.4` / `compressed-tensors==0.17.0` /
  `flashinfer-python==0.6.18` / `humming-kernels[cu13]` / `nvidia-cutlass-dsl[cu13]` /
  `quack-kernels` / `tilelang`（N-gram PLE 需要）/ `numba`。**别去手工钉这些版本**，让 resolver 决定，装完再核。
- `triton_kernels` 不用装（非硬依赖，见 §6.3）。
- **不要**显式 `VLLM_GDN_DECODE_KERNEL=cuda`（会让不满足条件的配置直接抛错而不回落 triton）。
- 新环境里 `transformers>=5.16.1` 只为**调优路径**服务；推理侧 vLLM 自带 `Qwen4ExpConfig`
  （`vllm/transformers_utils/config.py:128` → `vllm/models/qwen4_exp/config.py`），
  所以旧记录里"transformers 5.15.1 不认 qwen4_exp"只卡 tuning，不卡加载。
- 网络受限时先导出代理（本机 `HF_HOME=/models/huggingface` 已有 pile-10k / 各 bench 缓存，可离线评测）。

---

## 8. 风险与排查清单（本模型特有）

| 症状 | 根因 | 处置 |
|---|---|---|
| 引擎起不来：`mm_mxfp8 requires N >= 128, got N=96` | `in_proj_a/b` 被量化（融合后 N=96） | ignore `in_proj_a,in_proj_b`（M0/M1 已含）；跑 `audit_precision_plan.py` 会提前拦下 |
| `Invalid JSON: ...kernel_config={"moe_backend":"marlin"}`（lm_eval 启动即报） | 这版 lm_eval 的 `--model_args`（`_cli/utils.py:160-176` MergeDictAction）先对**第一个 token** 试 `json.loads`，串里含 `{` 就报错**不回退**——逗号 k=v 串里混进任何 dict 值必炸 | 脚本已改成**整体 JSON** 传 `--model_args`（int/float/bool/嵌套 dict 类型天然正确；已用 `try_parse_json` 单测 + 桩 lm_eval 确认它作为单个 argv 传入）。27B 脚本没有 dict 值，逗号串仍可用 |
| `NotImplementedError: Qwen4Exp QSA does not support KV quantization` | 稀疏注意力路径不支持 KV 量化 | 去掉 `kv_cache_dtype=fp8` / 产物里的 `kv_cache_scheme`；要这能力得上 nightly（#55557） |
| PLE 相关加载/精度异常 | PLE 只有"已序列化 FP8"这一条量化通路 | 保持 `ple` 整块 BF16（本文清单已如此） |
| 产物体积远大于预期（≈180 GB 而不是 60 GB） | PLE n-gram 表 102 GB 压不掉，占总参数 28.4% | 这是**结构决定**，不是配置错；要再降只能动 PLE（另议） |
| 显存装不下 / OOM | 102 GB 表必须常驻 + experts 权重 | M0 至少 TP2；bf16 baseline 用 TP4；`max_num_seqs` 先压到 64 |
| `--layer_config` 写 `act_bits:8` 想拿 W4A8 | llm_compressor 按有效 weight bits 重建整组 preset | 想要 W4A8 得换 scheme 思路（MX 系混合就是 W4A4 + W8A8 两组） |
| 抄了 27B 的清单 | 27B 是 dense，`mlp.gate` 禁写；本模型是 MoE，`mlp.gate` **必写** | 两套清单不要互抄（见附录 B） |
| 抄了 W4A16 先例的清单 | 那份把 `linear_attn,self_attn,indexer,shared_expert` 全保 BF16 | MX 系这些落在 MXFP8 是安全的（参考产物即如此），照抄会白白多留 2.9 GB |
| `RuntimeError: shape '[512, 2560, 10]' is invalid for input of size 15728640` | **vLLM bug**：MXFP8 MoE 的 FlashInfer TRTLLM scale swizzle 会把列补到 4 的倍数，之后却按未 pad 的形状 reshape；每卡列数 = `moe_intermediate/(TP·32)` = 20/TP，TP=2→10、TP=4→5 都会 pad（§6.6） | 走 M0（experts 用 MXFP4，绕开该函数）；或 M1 + TP=2 + `moe_backend=marlin`（已端到端验证）。**TP=1 也不行**：PLE n-gram 表要整份拷贝 95.37 GiB → OOM |
| `No available shared memory broadcast block found in 60 seconds`（+ 期间 GPU util 0%） | **不是错误**，是 vLLM 的 keepalive INFO：某进程在做耗时活（编译、CUDA graph 捕获、权重/KV 后处理）。512-expert MoE 捕获 83 个 shape 本就要好几分钟 | **耐心等**，别杀进程、更不要为此换 `enforce_eager=True`（本会话误判过一次，已纠正） |
| 磁盘写爆 | `/models` 已用 96%（剩 ~508 GB），而源模型 336 GB + M0 产物 ~180 GB | 量化前先 `df -h /models`；必要时产物落 `/`（overlay 剩 1.2 TB）或先清旧产物 |
| 想跑调优路径 | 180 B 全量加载 + 校准，先例 10 h 且**精度不如 RTN**（0.8800 vs 0.8819） | 除非 M0/M1 掉点超阈值，不要碰 |

## 9. 产物 / 文件清单

| 文件 | 用途 |
|---|---|
| `Qwen3.8-Flash-Next-quantization-recipe.md` | 本文 |
| `quant_Qwen3.8-Flash-Next-mxfp.sh` | M0/M1/M2 一键量化（内置前置审计门 + 导出核对 + `DRY_RUN`） |
| `eval_Qwen3.8-Flash-Next-quant.sh` | 冒烟 → 小 limit → 全量评测（`SMOKE_ONLY` / `TP` / `ENVBIN` 可调） |
| `setup_vllm_env_qwen38fn.sh` | 系统 python venv + uv 的干净环境构建（§7） |
| `audit_precision_plan.py` | 量化前审计（本轮已扩展：fused 3-D experts 展开、`shard_N` 归并、逻辑层数预测） |
| `check_quant_export.py` | 导出后自洽核对 |
| `smoke_vllm_ckpt.py` | vLLM 加载 + 生成冒烟（通用化，两个模型共用） |
| `/models/auto-quant/recipes/quant/Qwen__Qwen3.8-Flash-Next.json` | pinned 机器可读 recipe（本轮新增 MX 三档 + 纠正两处旧结论） |

## 附录 A · 复现本文数字

```bash
cd /models/lkk/cri_models
IGN="visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection"

python3 audit_precision_plan.py --model ./Qwen3.8-Flash-Next --plan-only-table          # §1.3
python3 audit_precision_plan.py --model ./Qwen3.8-Flash-Next --scheme MXFP8 \
  --ignore_layers "$IGN" --layer_config '{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}' # §2 / §5.1
python3 audit_precision_plan.py --model ./Qwen3.8-Flash-Next --scheme MXFP8 \
  --ignore_layers "$IGN"                                                                  # §5.2
```

## 附录 B · 与 Qwen3.8-27B 的方法差异（同族不同方子）

| | Qwen3.8-27B（dense） | Qwen3.8-Flash-Next（qwen4_exp MoE） |
|---|---|---|
| `mlp.gate` | **禁写**（误伤 `gate_proj`） | **必写**（router 必保） |
| experts | 无 | fused 3-D，auto-round 自动拆分 |
| 量化主体 | MLP 61.6% + GDN 20% | routed experts 67.1% |
| 压缩上限 | ~2.8×（全 FP4） | **~2.0×**（PLE n-gram 表 102 GB 封顶） |
| 特殊明文模块 | mtp、in_proj_a/b、conv1d | 上述 + hyper_connection、block_inject_weight、ple、indexer 之外全 MXFP8 |
| 调优路径 | 可行（24 GB 级显存） | 先例 10 h 且无收益，不建议 |
