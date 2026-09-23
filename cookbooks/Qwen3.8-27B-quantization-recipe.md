# Qwen3.8-27B 结构分析 + AutoRound MXFP8 / MXFP4 量化方案

- 模型：`/models/lkk/cri_models/Qwen3.8-27B`（bf16，18 分片，55.56 GB）
- 工具：auto-round 0.16.0（editable，源码 `/models/lkk/auto-round`）@ conda env `muse_verify`
- 推理/评测：`/models/lkk/muse_support/vllm`（v0.28.1rc1.dev312+g41848caa6）+ lm_eval
- Recipe 依据：`/models/auto-quant/recipes/schemes/{mxfp8,mxfp4}.yaml`、`/models/auto-quant/recipes/quant/Qwen__Qwen3.8-Flash-Next.json`、`/models/auto-quant/docs/{day0_quantization_recipe.md,autoround_full_reference.md}`、`/models/auto-quant/aqa/agent/recipes/auto_round_recipe.md`
- 本文所有数字均为**实测**（只读 `config.json` + safetensors header，不加载权重），复现命令见附录 A。

---

## 0. 结论速览

| 方案 | scheme | 精度分配 | 平均有效比特 | 产物体积 | 压缩比 | 定位 |
|---|---|---|---|---|---|---|
| **1 MXFP8-RTN** | `MXFP8` | 文本塔 Linear 全 MXFP8，embed/lm_head/visual/mtp/in_proj_a·b 保 BF16 | 8.99 | ≈32.0 GB | 1.74× | ✅ **已实跑 85 s + 冒烟通过**（§3.1），基准待跑；同类 dense 多模态 gemma-4-31B 上实测近无损（§7） |
| **2 MXFP4-Mixed** | `MXFP4` + `layer_config` | MLP=MXFP4，attention（GDN+full）=MXFP8，其余 BF16 | 6.53 | ≈23.4 GB | 2.37× | **主推的 FP4 方案** |
| **3 MXFP4** | `MXFP4` | 文本塔 Linear 全 MXFP4 | 5.49 | ≈19.8 GB | 2.80× | 激进档，须先冒烟 |
| 4 = 方案 2/3 + 调优 | 同上 + `--iters 200 --enable_alg_ext` | 同上 | 同上 | 同上 | 同上 | 方案 1–3 掉点超阈值时才付费跑 |

三条硬性结论（都有证据，别改）：

1. **`in_proj_a,in_proj_b` 必须保 BF16**。vLLM 把它们融合成 `in_proj_ba`（N=48+48=96），而 MXFP8 的 FlashInfer CuTe-DSL kernel 断言 `N >= 128`。现存的 `/models/lkk/cri_models/qwen3.8-27B-MXFP8-model-free` 就是漏了这条，**引擎启动即崩**（§3）。
2. **绝不能写 `mlp.gate`**。这是 dense 模型（无 routed experts），`mlp.gate` 子串会命中 `mlp.gate_proj`，把 5.7B 的 gate_proj 静默跳过。
3. **`--format llm_compressor`**（compressed-tensors，vLLM 直接加载）；MX 系 group_size 固定 32，本模型所有 Linear 的 `in_features`（5120/6144/10240/17408）都被 32 整除，无对齐障碍。

---

## 1. 模型结构分析

### 1.1 架构判定（以 config 为准，不看模型名）

| 项 | 值 | 说明 |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | 多模态条件生成（VL），非纯文本模型 |
| `model_type` | `qwen3_5` / `qwen3_5_text` | transformers 5.16.1 已注册（`CONFIG_MAPPING`），调优路径可用 |
| 语言塔 | dense，**无 MoE**（`mlp.gate_proj/up_proj/down_proj`，无 `experts`） | 与 Qwen3.5-35B-A3B 的关键差异 |
| 混合注意力 | `num_hidden_layers=64`，`full_attention_interval=4` → **48 层 linear_attention + 16 层 full_attention**（层号 3,7,…,63） | linear 层是 Gated DeltaNet（GDN）风格 |
| hidden / inter | 5120 / 17408，`hidden_act=silu`（SwiGLU） | |
| 注意力维度 | `head_dim=256`，24 Q heads / 4 KV heads，`attn_output_gate=true`（q_proj 输出 12288=2×24×256，一半是 gate），`partial_rotary_factor=0.25`，QK-norm | |
| GDN 维度 | `linear_num_key_heads=16×128`、`linear_num_value_heads=48×128`、`linear_conv_kernel_dim=4` | in_proj_qkv 输出 10240=2·16·128+48·128 |
| 词表 / 上下文 | 248320 / 262144 | embed 与 lm_head **各 1.27B**（不共享权重，`tie_word_embeddings=false`） |
| 视觉塔 | `depth=27`，hidden 1152，`patch_size=16`，`spatial_merge_size=2`，`out_hidden_size=5120`，merger 2 层 MLP | 460.7M 参数 |
| MTP | `mtp_num_hidden_layers=1`（1 层 full-attn + 1 层 dense MLP + `mtp.fc` 5120×10240） | 424.7M，投机解码用 |
| 权重 dtype | 全部 BF16（1199 个张量） | |

### 1.2 模块 / 张量清单（实测 shape）

**语言塔 `model.language_model.*`**

| 模块 | 出现次数 | 张量（shape） | 单层参数 |
|---|---|---|---|
| `embed_tokens` | 1 | `[248320, 5120]` | 1271.4M |
| `layers.N.input_layernorm` / `post_attention_layernorm` | 64 / 64 | `[5120]` | 10K |
| **linear_attn**（GDN，48 层） | 48×8 | `in_proj_qkv [10240,5120]`、`in_proj_z [6144,5120]`、`in_proj_a [48,5120]`、`in_proj_b [48,5120]`、`out_proj [5120,6144]`、`conv1d [10240,1,4]`、`norm [128]`、`A_log [48]`、`dt_bias [48]` | 383.3M（含 MLP） |
| **self_attn**（full，16 层） | 16×6 | `q_proj [12288,5120]`、`k_proj [1024,5120]`、`v_proj [1024,5120]`、`o_proj [5120,6144]`、`q_norm [256]`、`k_norm [256]` | 372.3M（含 MLP） |
| **mlp**（dense SwiGLU，64 层） | 64×3 | `gate_proj [17408,5120]`、`up_proj [17408,5120]`、`down_proj [5120,17408]` | 267.4M |
| `norm` | 1 | `[5120]` | — |

**视觉塔 `model.visual.*`**：`patch_embed.proj`（3D 卷积）、`pos_embed [2304,1152]`、27×`blocks`（`attn.qkv [3456,1152]`+bias、`attn.proj [1152,1152]`+bias、`mlp.linear_fc1 [4304,1152]`、`mlp.linear_fc2 [1152,4304]`、`norm1/norm2`）、`merger.{norm,linear_fc1 [4608,4608],linear_fc2 [5120,4608]}`。

**MTP `mtp.*`**：`pre_fc_norm_embedding`、`pre_fc_norm_hidden`、`fc [5120,10240]`、`layers.0.{self_attn(q/k/v/o + q_norm/k_norm), mlp(gate/up/down), 2×norm}`、`norm`。

**输出头** `lm_head [248320,5120]`。

### 1.3 参数量与占比（总计 **27.781 B**，bf16 55.56 GB）

| 模块 | 参数 | 占比 | bf16 体积 |
|---|---|---|---|
| LM MLP（dense SwiGLU，64 层） | 17 112.8 M | **61.60 %** | 34.23 GB |
| LM linear_attn（GDN，48 层） | 5 562.1 M | **20.02 %** | 11.12 GB |
| LM self_attn（full，16 层） | 1 677.7 M | 6.04 % | 3.36 GB |
| `embed_tokens` | 1 271.4 M | 4.58 % | 2.54 GB |
| `lm_head` | 1 271.4 M | 4.58 % | 2.54 GB |
| Vision tower | 460.7 M | 1.66 % | 0.92 GB |
| MTP | 424.7 M | 1.53 % | 0.85 GB |
| LM norms / 其它 | 0.7 M | 0.00 % | 0.01 GB |

要点（直接决定量化预算分配）：

- 量化收益几乎全在 **MLP（61.6%）+ GDN 投影（20.0%）**；两者合计 81.6%。
- **embedding + lm_head = 9.15%** 且常规做法保 BF16 —— 这是压缩比的天花板：即使其余全 FP4，整模型也不可能低于 ~5.5 bit/参数（实测见 §4）。
- 全模型 2D Linear 权重（含 embed/lm_head/visual/mtp）占 95.4%，但真正值得下刀的是文本塔的 24.3 B。
- 视觉塔 1.66% + MTP 1.53% 保明文只多占 ~2.4 GB，换掉两类兼容性风险。

---

## 2. auto-round 会怎么处理这个模型

### 2.1 两条路径与本机环境

| | `--model_free`（RTN，免校准） | 调优（`--iters 200`） |
|---|---|---|
| 是否加载模型 | 否，流式重写 safetensors 分片 | 是（transformers 必须认 `qwen3_5`：**5.16.1 认**） |
| 校准数据 | 不需要 | 默认 `NeelNanda/pile-10k`，本机已在 `HF_HOME=/models/huggingface` 缓存 |
| 耗时（本机实测） | **85.36 秒**（2026-09-21 重跑，峰值 RAM 3.28 GB；纯 I/O 受限：55 GB 读 + 30 GB 写） | 数小时级（64 层 ×200 iter 前向/反向） |
| 适用 | MX 系 + 大模型建下限 | 掉点超阈值时追精度 |

`MXFP8`/`MXFP4` 都在 model_free 白名单内（`data_type="mx_fp"`，`bits ∈ {4,8}`，`group_size=32`，见 `/models/lkk/auto-round/auto_round/compressors/model_free.py:42`）。纪律：**跑完 grep 日志确认没有 `falling back to the regular AutoRound flow`**（静默回退=其实走了全量加载，day-0 架构上直接炸）。

### 2.2 自动处理（不用手写）

- **非 2D 张量**（`input_layernorm`、`post_attention_layernorm`、`linear_attn.norm`、`q_norm/k_norm`、`A_log`、`dt_bias`、`conv1d`）不参与 MX 打包。
- **预定义 skip**：`_BLOCK_NAME_TO_IGNORE = ("shared_expert_gate.", ".gate.", "embed", "conv")`（`/models/lkk/auto-round/auto_round/utils/model_free_utils.py:42`）。注意 `.gate.` 带尾点，**不会**命中 `mlp.gate_proj`（实测产物里 gate_proj 确实被量化了）。
- **VLM 默认只量化文本塔**、**lm_head 默认不量化**（除非 `--quant_lm_head`）。
- `--ignore_layers` 是**子串/glob 转正则**匹配（`to_standard_regex`：无正则元字符 → `.*pattern.*`），匹配对象是**层名**（张量名去掉最后的 `.weight`）。

### 2.3 必须显式保护的层（本模型特有）

| 层 | 参数 | 为什么要保 |
|---|---|---|
| `linear_attn.in_proj_a` / `in_proj_b` | 23.6 M | vLLM `qwen3_5.py:224-225` 把两者融合为 `in_proj_ba`（N=96）；`FlashInferCutedslMxfp8LinearKernel` 断言 `mm_mxfp8 requires N >= 128`（`kernels/linear/mxfp8/flashinfer.py:148-158`）→ **profile_run 阶段引擎直接崩**。MXFP4 的 CUTLASS kernel 会把 N 向上补到 128（`kernels/linear/mxfp4/flashinfer.py:40-46`）不崩，但为保持两档一致 + 省 0.04% 参数，同样保 BF16 |
| `visual`（全塔） | 460.7 M | 纯文本评测不走；量化后 mm 路径无对应 kernel |
| `lm_head` / `embed_tokens` | 2 542.8 M | 输出层对低比特最敏感，业界 100% 保高精度 |
| `mtp` | 424.7 M | 投机解码侧模块，主路径不受益；先验 recipe（Flash-Next）也 ignore 它 |
| ~~`mlp.gate`~~ | — | **禁止**：dense 模型上会误伤 `mlp.gate_proj`（`Qwen3.8-27B_MXFP4` 分析记录里踩过） |

---

## 3. 现存 MXFP8 产物复核（失败根因，已定位）

`/models/lkk/cri_models/qwen3.8-27B-MXFP8-model-free`（由 `quant_Qwen3.8-27B.sh` 产出，`--ignore_layers "visual,lm_head,embed_tokens"`）：

- **量化本身是成功的**：权重 28.4 GB（`du` 30G），`format: mxfp8-quantized` / `compressed-tensors` / W8A8 g32 / `scale_dtype: torch.uint8`（E8M0），`ignore` 162 项 = `lm_head`(1) + `embed_tokens`(1) + `visual`(112) + `linear_attn.conv1d`(48)；`check_quant_export.py` 对它是通过的 —— 也就是说**配置自洽，问题纯在 kernel 维度**。
- **vLLM 加载即崩**（`mxp8-Qwen3.8-27B.log:63`）：

  ```
  torch._dynamo.exc.ObservedAssertionErrorError: raised exception
  AssertionError('mm_mxfp8 requires N >= 128, got N=96. out_features is too small for mm_mxfp8.')
  ```

  崩溃点在 `determine_available_memory → profile_run → _dummy_run`，即第一次前向就把所有 MXFP8 Linear 都过了一遍 kernel —— **这类维度不兼容必在冒烟阶段暴露**，不需要跑完整评测。
- 根因即 §2.3 第一条：`in_proj_a/in_proj_b` 未进 ignore（它们**确实被量化了**，产物里带 `weight_scale`）。
- 处置：该目录**不可用**，用 §5 方案 1 重新生成（另起 `Qwen3.8-27B-MXFP8-AutoRound`，别覆盖旧目录，留作证据）。附带发现：该产物把 `mtp.*` 也量化了，重跑时一并 ignore。

### 3.1 修正后的 MXFP8 重跑（2026-09-21 08:40，已完成）

`bash quant_Qwen3.8-27B-mxfp.sh mxfp8` → `/models/lkk/cri_models/Qwen3.8-27B-MXFP8-AutoRound`：

| 项 | 实测 |
|---|---|
| 耗时 / 峰值 RAM | **85.36 s** / 3.28 GB（真免加载，日志无 `falling back` 字样） |
| 产物 | 30 GB，18 分片，`format=mxfp8-quantized`，`group_0: W8A8 g32` |
| 量化层 | **400** = `mlp.{gate,up,down}_proj`×64 (192) + `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`×48 (144) + `self_attn.{q,k,v,o}_proj`×16 (64) —— 与 §4 审计表逐项吻合 |
| 明文层 | **266 项 ignore**：`lm_head`、`embed_tokens`、`in_proj_a`×48、`in_proj_b`×48、`conv1d`×48、visual 112、mtp 8（合计 2D Linear 617 = 400 量化 + 217 明文） |
| 导出核对 | `check_quant_export.py` → OK（明文全在 ignore 内，量化层全被 config_groups 覆盖） |
| **vLLM 加载冒烟** | **PASS**（`smoke_MXFP8-AutoRound.log`）：`Using FlashInferCutedslMxfp8LinearKernel` → 加载 28.35 GiB / 24.6 s → **`Initial profiling/warmup run took 5.60 s` 过了**（旧产物正是在这步崩 `N >= 128`）→ 3 条确定性生成语义正确（"The capital of France is"→`Paris…`；"4+4="→`8. 8+8=16…`；`def quicksort(arr):`→可运行实现）→ `SMOKE OK` |

---

## 4. 精度分配（量化前必须过一遍，别靠猜）

用 `audit_precision_plan.py`（附录 A）把候选命令的 ignore/layer_config 逐张量跑一遍匹配，产出分配表并校验 vLLM kernel 约束。下面三张表就是三条命令的结果。

**方案 1 · MXFP8**（`--scheme MXFP8 --ignore_layers visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b`）

| 位宽 | 参数 | 占 Linear |
|---|---|---|
| 8 bit（MLP 17.11 B + GDN 5.54 B + full-attn 1.68 B） | 24.327 B | 87.6 % |
| 16 bit（embed+lm_head 2.54 B、visual 0.44 B、mtp 0.41 B、in_proj_a/b 0.024 B） | 3.450 B | 12.4 % |

→ ≈32.0 GB，1.74×，平均 8.99 bit。审计校验：**OK**。

**方案 2 · MXFP4-Mixed**（`--scheme MXFP4` + `--layer_config` 把 `self_attn`/`linear_attn` 提到 8 bit）

| 位宽 | 覆盖模块 | 参数 | 占比 |
|---|---|---|---|
| 4 bit | `mlp.{gate,up,down}_proj` ×64 | 17.113 B | 61.6 % |
| 8 bit | `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`、`self_attn.{q,k,v,o}_proj` | 7.214 B | 26.0 % |
| 16 bit | 同上明文清单 | 3.450 B | 12.4 % |

→ ≈23.4 GB，2.37×，平均 6.53 bit。审计校验：**OK**。
（`layer_config` 的正则键命中不了已 ignore 的层——ignore 先生效，所以 `mtp.*.self_attn.*` 与 `in_proj_a/b` 仍是 BF16，表里已验证。）

**方案 3 · MXFP4 全量**（同方案 1 的 ignore，`--scheme MXFP4`）

4 bit 24.327 B（87.6%）+ 16 bit 3.450 B（12.4%）→ ≈19.8 GB，2.80×，平均 5.49 bit。审计校验：**OK**。

**方案 0 · 现存失败配置**（只 ignore `visual,lm_head,embed_tokens`）：审计器准确复现了崩溃——报出 96 个 `in_proj_a/in_proj_b` 张量 `N=48 < 128`。也就是说这个坑**在敲命令前就能被拦下**，不必等 vLLM 起引擎。

---

## 5. 量化命令

### 5.0 公共前置

```bash
source /models/miniforge3/etc/profile.d/conda.sh
conda activate muse_verify          # 或直接 export PATH=/models/miniforge3/envs/muse_verify/bin:$PATH
export HF_HOME=/models/huggingface
export AR_MODEL_FREE_SHARD_PARALLELISM=1     # 控制 model_free host RAM 峰值（55GB 分片流式重写）
nvidia-smi --query-gpu=index,memory.used --format=csv   # 铁律：不与他人抢卡，确认空闲再动
```

统一清单（**方案 1/2/3 共用**）：

```
IGNORE="visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b"
```

### 5.1 方案 1 — MXFP8（先跑，建下限）

```bash
auto-round \
  --model_name /models/lkk/cri_models/Qwen3.8-27B \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b \
  --format llm_compressor \
  --device_map auto \
  --output_dir /models/lkk/cri_models/Qwen3.8-27B-MXFP8-AutoRound
```

### 5.2 方案 2 — MXFP4-Mixed（MLP FP4 + attention FP8，主推 FP4 档）

```bash
auto-round \
  --model_name /models/lkk/cri_models/Qwen3.8-27B \
  --model_free \
  --scheme MXFP4 \
  --layer_config '{"self_attn":{"bits":8,"data_type":"mx_fp","group_size":32,"act_bits":8,"act_data_type":"mx_fp","act_group_size":32},"linear_attn":{"bits":8,"data_type":"mx_fp","group_size":32,"act_bits":8,"act_data_type":"mx_fp","act_group_size":32}}' \
  --ignore_layers visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b \
  --format llm_compressor \
  --device_map auto \
  --output_dir /models/lkk/cri_models/Qwen3.8-27B-MXFP4-Mixed-AutoRound
```

导出应为 `format: mixed-precision` + 两个 `config_groups`（W4A4 与 W8A8）——同架构先例 `Qwen3.5-35B-A3B-MXFP4-Mixed` 就是这个形状。

### 5.3 方案 3 — MXFP4 全量（激进）

```bash
auto-round \
  --model_name /models/lkk/cri_models/Qwen3.8-27B \
  --model_free \
  --scheme MXFP4 \
  --ignore_layers visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b \
  --format llm_compressor \
  --device_map auto \
  --output_dir /models/lkk/cri_models/Qwen3.8-27B-MXFP4-AutoRound
```

### 5.4 方案 4 — 调优路径（只在 1–3 掉点超阈值时付费）

去掉 `--model_free`，加校准/优化参数（其余同上）：

```bash
auto-round \
  --model_name /models/lkk/cri_models/Qwen3.8-27B \
  --scheme MXFP4 \
  --layer_config '<同 5.2>' \
  --ignore_layers visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b \
  --iters 200 --nsamples 128 --seqlen 2048 --enable_alg_ext \
  --gradient_accumulate_steps 8 --batch_size 1 --low_gpu_mem_usage \
  --device_map auto --format llm_compressor \
  --output_dir /models/lkk/cri_models/Qwen3.8-27B-MXFP4-Mixed-AutoRound-Tuned
```

调优路径注意：① 需要 torch.compile ×多卡 hook 的上游坑（pytorch#185372）时加 `--disable_torch_compile` 保底；② 导出后经 transformers 重序列化，**必须核对 config 结构没漂移**（`full_attention` 之类字段被规范化会让 vLLM 拒收）；③ 大模型上调优不一定更好（Flash-Next 实测 iters=200 比 RTN 还差 0.95pp mmlu）。

### 5.5 一键脚本

```bash
cd /models/lkk/cri_models
./quant_Qwen3.8-27B-mxfp.sh mxfp8          # 方案1
./quant_Qwen3.8-27B-mxfp.sh mxfp4_mixed    # 方案2
./quant_Qwen3.8-27B-mxfp.sh mxfp4          # 方案3
./quant_Qwen3.8-27B-mxfp.sh mxfp4_mixed tune   # 方案4
DRY_RUN=1 ./quant_Qwen3.8-27B-mxfp.sh mxfp8    # 只跑前置审计 + 打印最终命令，不动 GPU
```

脚本流程：前置审计（不过则 `exit 2`，压根不发命令）→ 打印可复制的 auto-round 命令 → 跑量化 →
①grep model_free 静默回退 ②`check_quant_export.py` 核对导出与落盘权重自洽。日志落
`quant_<产物名>.log`，审计表落 `audit_<产物名>.log`。

> v1 脚本有个坑已修：`SUFFIX="$( [ cond ] && echo -Mixed )"` 在 `set -e` 下命令替换返回 1，
> 脚本走到那行就**静默退出**（现象是只打了审计就没了，误以为脚本坏了而把 `set -euo pipefail` 注释掉）。
> 现在改成显式 `if`，**请保持 `set -euo pipefail` 处于打开状态**。

---

## 6. 验证与评测流程（成本递增，任一步不过就不往下）

```bash
# ① 导出自洽性核对：明文 Linear 是否全部被 ignore 覆盖、量化层是否全部被 config_groups 覆盖
#    （不一致 = vLLM 建层 KeyError；已用现存两个产物验证过脚本本身）
python check_quant_export.py <OUT>

# ② 只做 vLLM 加载 + 生成冒烟（N>=128 这类 kernel 维度问题在 profile_run 就暴露，几分钟）
SMOKE_ONLY=1 ./eval_Qwen3.8-27B-quant.sh <OUT>
python smoke_vllm_ckpt.py <OUT> 1        # 也可单独跑（必须是独立脚本，见 §8 的 <stdin> 坑）

# ③ 小 limit 基准（piqa limit 16）  ④ 全量基准 + 与 bf16 baseline 对比 gate
TASKS=piqa LIMIT=16 ./eval_Qwen3.8-27B-quant.sh <OUT>
./eval_Qwen3.8-27B-quant.sh <OUT>          # 正式跑：只跑 piqa,mmlu,hellaswag（默认不冒烟、不跑 gsm8k）
RUN_SMOKE=1 ./eval_Qwen3.8-27B-quant.sh <OUT>   # 想顺带先冒烟（会多完整加载一遍）
RUN_GSM8K=1 ./eval_Qwen3.8-27B-quant.sh <OUT>   # gsm8k 单列开关（口径要 --apply_chat_template 多轮）
```

产物路径给相对路径也能跑（脚本内 `realpath` 归一化），但建议直接给绝对路径——lm_eval 的
结果目录名取自该字符串，带 `/` 或相对前缀会让结果目录名很难认。默认用 `CUDA_VISIBLE_DEVICES=3`，
跑前自己 `nvidia-smi` 复核该卡空闲。

评测口径与现有 `eval_Qwen3.8-27B-mxfp8.sh` 一致：`language_model_only=True`、`enable_thinking=False`、`reasoning_parser=qwen3`、`seed=42`、gsm8k 走 `--apply_chat_template --fewshot_as_multiturn --batch_size 64`。**baseline 必须先跑**（本机 `lm_eval_results/` 里还没有 Qwen3.8-27B 的记录，否则没有对照）。

---

## 7. 精度预期（本机同栈实证，不是模型口径宣称）

| 模型（均 `language_model_only` / lm_eval / seed 42） | mmlu | piqa(acc) | hellaswag(acc) | gsm8k |
|---|---|---|---|---|
| gemma-4-31B bf16（dense 多模态，结构最接近本模型） | 0.8113 | 0.8292 | 0.6593 | 0.8666 |
| gemma-4-31B **MXFP8** | 0.8130 | 0.8254 | 0.6586 | 0.8688 |
| gemma-4-26B-A4B bf16 | 0.7446 | 0.8243 | 0.6341 | 0.7384 |
| gemma-4-26B-A4B **MXFP8** | 0.7455 | 0.8237 | 0.6345 | 0.7248 |
| Qwen3.5-35B-A3B bf16 | 0.8280 | 0.8243 | 0.6233 | 0.9052 |
| Qwen3.5-35B-A3B **MXFP4-Mixed**（experts FP4 + 其余 FP8） | 0.8244 | 0.8128 | 0.6151 | 0.9583 |
| gemma-4-26B-A4B **纯 MXFP4**（model_free） | 0.2300 | 0.4924 | 0.2508 | 0.0000 |

读法：

- **MXFP8 = 近无损**（Δ ≤ 0.4pp，gsm8k 甚至 +0.2pp）。方案 1 预期直接 GO，gate 建议 `max_accuracy_drop: 0.01`。
- **Mixed（FP4 主体 + FP8 敏感层）：知识类/常识类掉 0.4~1.2pp**（mmlu −0.36pp、piqa −1.15pp、hellaswag −0.82pp）；该对照里 gsm8k 反而 +5.3pp，两次 run 的 chat/fewshot 口径是否完全一致未查明，**不要把这条当结论**，只当"mixed 档总体在 1~2pp 内"的旁证。
- **纯 model_free MXFP4 出过随机分数的产物**（gemma-4-26B-A4B-MXFP4；同目录另有 `fix_mxfp4_moe_scale_padding.patch`，那次是 MoE 场景的 scale padding 问题）。本模型是 **dense，不存在 fused-expert 路径，风险显著更低**，但**必须先过冒烟 + 小 limit** 再上全量，别把随机分数当"精度掉这么多"。
- 27.8B dense 的 FP4 敏感度介于两者之间；MLP 占 61.6% 且 dense MLP 是最耐量化的部分，attention 提到 MXFP8（只占 26% 参数）是性价比最高的一档保险。

---

## 8. 风险与排查清单

| 症状 | 根因 | 处置 |
|---|---|---|
| `line 84: --model_name: command not found` | 命令数组首元素漏了程序名 `auto-round`，`"${CMD[@]}"` 把第一个参数当命令执行 | 已在 v2 修正（`CMD=(auto-round ...)`）；用 `DRY_RUN=1` 可先看命令再执行 |
| 脚本只打了审计表就结束 / 没跑量化 | v1 的 `SUFFIX="$( [ ] && echo )"` 命令替换在 `set -e` 下返回 1，静默终止 | 已改显式 `if`；不要把 `set -euo pipefail` 注释掉，用 `DRY_RUN=1` 验证命令 |
| `Engine core initialization failed` + `FileNotFoundError: .../<stdin>` | 冒烟代码用 `python - <<PY` 从 stdin 喂，vLLM 的 spawn 子进程要重新 exec `__main__`，找不到 `<stdin>` —— **与产物无关** | 冒烟必须是独立脚本且入口包在 `if __name__ == "__main__"`（已落 `smoke_vllm_ckpt.py`） |
| `mm_mxfp8 requires N >= 128, got N=96` | `in_proj_a/b` 被量化（vLLM 融合成 N=96） | ignore `in_proj_a,in_proj_b`，重跑 |
| vLLM 建层 `KeyError: ...weight_scale` | 导出 `ignore` 没覆盖所有明文 Linear | 只靠 `--ignore_layers` 显式列全，跑 §6① 核对 |
| 分数随机（mmlu≈0.25） | 产物坏了，不是精度问题 | 先冒烟 + `LIMIT=16`；比对 groups/ignore；MXFP4 走 mixed 兜 |
| 日志 `falling back to the regular AutoRound flow` | `--model_free` 与 scheme/format 组合不被支持，静默转全量加载 | 检查 scheme 是否 MX 系 + `llm_compressor`；必要时去掉 model_free |
| `mlp.gate` 想保护路由却把 `gate_proj` 跳了 | 子串匹配（本模型无 router） | dense 模型不要写 `mlp.gate`；真要精确锚定用 `mlp.gate$` |
| 调优路径 `AutoConfig` 不认 `qwen3_5` | transformers 版本过旧 | 用 5.16.1+（`muse_verify` 已满足）；或退回 model_free |
| 多卡 torch.compile 崩（pytorch#185372） | 上游 bug | 加 `--disable_torch_compile` |
| host RAM 打爆 | 大分片并发重写 | `AR_MODEL_FREE_SHARD_PARALLELISM=1`（已在脚本里） |
| `No module named 'triton_kernels.matmul_ogs'` 报错行 | 该栈的 MXFP4 triton 后端不可用 | 非致命：Blackwell 上走 FlashInfer CuTe-DSL 后端（日志有 `Using FlashInferCutedslMxfp8LinearKernel`）；忽略即可 |

---

## 9. 产物与命名

- 目录统一落 `/models/lkk/cri_models/`，命名 `<Model>-<Scheme>[-Mixed][-Tuned]-AutoRound`（对齐 `auto-quant/AGENTS.md` §6.3 的 artifact 命名规范；model_free/iters=0 在榜单口径里对应 `-RTN` 后缀）。
- 每个产物旁保留对应 `quant_<name>.log`；评测结果进 `lm_eval_results/`。
- 机器可读 recipe 已 pin 到 `/models/auto-quant/recipes/quant/Qwen__Qwen3.8-27B.json`（单一真源，后续 run 直接读它，不重推导）。
- 本次新增的工具脚本：`audit_precision_plan.py`（量化前审计）、`check_quant_export.py`（导出核对）、`smoke_vllm_ckpt.py`（vLLM 加载+生成冒烟）、`quant_Qwen3.8-27B-mxfp.sh`、`eval_Qwen3.8-27B-quant.sh`。

## 附录 A — 复现本文所有数字

```bash
cd /models/lkk/cri_models
# 模块参数占比表（§1.3）
python audit_precision_plan.py --model ./Qwen3.8-27B --plan-only-table

# 精度分配 + vLLM kernel 约束校验（§4，每条方案对应一次）
python audit_precision_plan.py --model ./Qwen3.8-27B --scheme MXFP8 \
  --ignore_layers visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b
python audit_precision_plan.py --model ./Qwen3.8-27B --scheme MXFP4 \
  --ignore_layers visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b \
  --layer_config '{"self_attn":{"bits":8,"data_type":"mx_fp"},"linear_attn":{"bits":8,"data_type":"mx_fp"}}'
```

审计器语义取自 auto-round 源码（`model_free_utils.py:42` 预定义 skip、`utils/common.py:818` 子串转正则、`_PatternMatcher.should_ignore/should_skip/resolve_scheme` 的优先级），并用现存 MXFP8 产物的真实 `ignore` 清单校验过一致。

## 附录 B — 关键代码位置

- vLLM MXFP8 kernel 维度断言：`/models/lkk/muse_support/vllm/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:148-158`
- vLLM MXFP4 kernel 的 N 补齐：`.../kernels/linear/mxfp4/flashinfer.py:40-46`
- `in_proj_b/a → in_proj_ba` 融合：`.../model_executor/models/qwen3_5.py:224-225, 310, 483`
- auto-round model_free 预定义 skip：`/models/lkk/auto-round/auto_round/utils/model_free_utils.py:42`
- auto-round 层名匹配：`/models/lkk/auto-round/auto_round/utils/common.py:818`
- 崩溃现场：`/models/lkk/cri_models/mxp8-Qwen3.8-27B.log:62-63`
