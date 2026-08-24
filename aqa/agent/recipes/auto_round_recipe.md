<!-- Distilled auto_round MXFP4/MoE recipe knowledge. Ported into aqa from
local_dispatch/usages/auto_round_mxfp4_summary.md; embedded into the agent-optimize
prompt so recommendations are grounded in real-world recipes, not guesswork. -->

# Auto-Round 量化工具调研与 MXFP4 量化 Recipe 总结

> 调研日期：2026-07-21。来源：intel/auto-round GitHub（main 分支）+ HuggingFace 上 25+ 个实际 MXFP4 模型仓库的 quantization_config / README。

## 1. Auto-Round 工具概览

### 1.1 量化算法
- **AutoRound（SignRound）**：对每个权重的舍入引入可学习参数 V（决定 round up/down），用 sign-gradient descent 逐 block 优化舍入方向，同时优化 min-max 裁剪系数。显存需求低（7B 单卡约 10 分钟），2–4 bit 精度 SOTA。
- **五种内置 Recipe**（`auto-round` CLI 直接指定）：

| Recipe | iters | nsamples | lr | 说明 |
|---|---|---|---|---|
| `auto-round`（默认） | 200 | 128 | 1/iters | 精度/成本平衡，推荐 W4A16 |
| `auto-round-best` | 1000 | 512 | 1/iters | 最佳精度，慢 4–5×，配 `enable_alg_ext` 推荐 W2A16 |
| `auto-round-light` | 50 | 128 | 5e-3 | 2–3× 加速 |
| `auto-round-opt-rtn` | 0 | 128 | — | 免校准，仍做 scale/zp 搜索 |
| `auto-round-rtn` | 0 | 0 | — | 纯 RTN，最快最省显存 |

- AutoGPTQ / AutoAWQ 在 auto-round 中只是**导出格式**（`--format auto_gptq/auto_awq`），算法仍是 sign-round 调优；混合 bit 建议用 `auto_round` 格式。

### 1.2 支持的数据类型
W2/W3/W4/W8A16、FP8（STATIC / BLOCK）、**MXFP4 / MXFP4_RCEIL / MXFP8 / NVFP4**、GGUF 全系列等。

**MXFP4 scheme 定义**（OCP MX 标准）：
```python
bits=4, group_size=32, data_type="mx_fp",
act_bits=4, act_data_type="mx_fp", act_group_size=32, act_sym=True, act_dynamic=True
```
- group_size 固定 32，scale 为 **E8M0**（uint8 指数）。
- 2025/09 起支持；`enable_alg_ext=True` 可提升 MXFP4 精度。
- **导出格式**：官方推荐 `llm_compressor`（compressed-tensors `mxfp4-pack-quantized`，vLLM 可直接加载）；`auto_round` 原生格式供研究；`fake` 仅供研究；`--model_free` 模式免校准直接输出 compressed-tensors。
- 推理硬件要求：vLLM/transformers 跑 MXFP4 需 compute capability ≥ 9.0（H100/B200）或 Triton ≥ 3.4 + triton_kernels。

### 1.3 关键 API 参数

```python
from auto_round import AutoRound
ar = AutoRound(
    model_name_or_path,
    scheme="MXFP4",            # 或 W4A16 / MXFP4_RCEIL / NVFP4 / dict
    iters=200,                 # 0 = RTN
    seqlen=2048, nsamples=128, batch_size=8,
    dataset="NeelNanda/pile-10k",
    enable_minmax_tuning=True,
    enable_alg_ext=False,      # 实验性增强，MXFP4/W2 提升明显
    low_gpu_mem_usage=False,   # 省 ~20G 显存，慢 20–30%
    quant_lm_head=False,
    layer_config={...},        # 按层覆盖
)
ar.quantize_and_save(output_dir, format="llm_compressor")
```

**layer_config**（支持完整名/模糊/正则匹配；bits≥16 即保持原精度）：
```python
layer_config = {
    "model.decoder.layers.*k_proj": {"bits": 8, "group_size": 32},
    "lm_head": {"bits": 16},     # 等价于忽略
    "lm_head": "GGUF:Q6_K",      # 也可直接给 scheme 字符串
}
```
- `fp_layers="lm_head,mlp.gate,self_attn"`（逗号分隔模糊匹配）是 INC 官方 recipe 保持层为 FP16 的惯用参数。
- 注意：vLLM/SGLang 会融合 MoE expert 与 QKV，**不建议给这些层设不同 bit 宽度**。

### 1.4 MoE 模型的自动处理
- auto-round 的 `special_model_handler` 会**自动把 router/gate（以 `.gate` 结尾的模块）加入预定义 ignore 列表**；DeepSeek 系还自动忽略 `moe.gate`、`eh_proj/shared_head`（MTP）等。
- 同一 transformer block 内所有 expert 的 gate/up/down 自动共享同一 scheme，无需手动配置。
- vLLM 在线 `quantization="mxfp4"` 对 MoE 有 bug——**MoE MXFP4 应离线预量化并以 llm_compressor 格式加载**。

## 2. HuggingFace 上 MXFP4 + AutoRound 模型盘点（25+ 个）

> 注意：Intel org 下只有 recipe 卡（如 `Intel/Qwen3-235B-A22B-mxfp4-AutoRound`、`Intel/Llama-4-Scout-17B-16E-Instruct-MXFP4-AutoRound-Recipe`），实际权重托管在 INCModel / INCModel2 / weiweiz1 / lvkaokao / Yi30 等账号。

| 模型 | 基模型 | 类型 | 配置要点 |
|---|---|---|---|
| INCModel/Qwen3-235B-A22B-MXFP4-AutoRound | Qwen3-235B | MoE | W4A4 g32，act mx_fp_rceil；94 层 `mlp.gate` 保 16bit；llm_compressor |
| INCModel/Qwen3-235B-A22B-MXFP4-FP8KV-FP8ATTN-AutoRound | 同上 | MoE | 同上 + FP8 静态 attention/KV；iters=0 |
| INCModel/DeepSeek-R1-MXFP4-AutoRound | DeepSeek-R1 | MoE(MLA) | iters=0（RTN）；全部 61 层 self_attn 保 16bit |
| INCModel/Qwen3-235B-A22B-Instruct-2507-MXFP4-CT-AutoRound | Qwen3-2507 | MoE | compressed-tensors；ignore 473 项（gate×94 + self_attn + lm_head） |
| INCModel/Kimi-K2.6-MXFP4-AutoRound | Kimi-K2.6 | MoE | **RTN**；`--ignore_layers shared_experts,self_attn,mlp.gate_proj,mlp.up_proj,mlp.down_proj` → 只量化 routed experts；精度保持 99.54% |
| INCModel/Qwen3-30B-A3B-AutoRound-MXFP4-W4A4 | Qwen3-30B-A3B | MoE | iters=0；self_attn + mlp.gate 全部保 16bit |
| INCModel2/GLM-5.2-MXFP4-Mixed-CT-AutoRound | GLM-5.2 | MoE | mixed：experts FP4 + 敏感层 FP8；ignore layernorm/router/indexer |
| INCModel2/Hy3-MXFP4-Mixed-CT-AutoRound | Hy3 | MoE | mixed FP4+FP8；ignore norm、`mlp.router.gate`、lm_head、embedding |
| xwu-intel/DeepSeek-V4-Flash-MXFP4-Mixed-CT-AutoRound | DeepSeek-V4 | MoE | ignore `ffn.gate`、attn compressor/indexer、embed、head |
| weiweiz1/DeepSeek-V2-Lite-MXFP4-autoround（& -RTN） | DS-V2-Lite | MoE | 旧版纯权重 mx_fp4e2m1；iters=50, nsamples=512, bs=4 |
| LeaderboardModel1/gpt-oss-20b-AutoRound-MXFP4-RTN | gpt-oss-20b | MoE | 全部 self_attn + mlp.gate 保 16bit |
| LeaderboardModel1/Qwen3.5-35B-A3B-AutoRound-MXFP4-ModelFree | Qwen3.5-A3B | MoE | model-free；ignore lm_head/embed/gate/shared_expert_gate/linear_attn |
| INCModel/Qwen3-1.7B、lvkaokao/Qwen3-0.6B/1.7B/32B、Doubleword/Qwen3-4B 等 | Qwen dense | Dense | W4A4 g32 mx_fp；部分 ignore self_attn，部分全量量化；Doubleword 用 seqlen=1024 |

### 官方 recipe（intel/neural-compressor examples）
- **MoE（Qwen3-235B/30B-A3B）**：`scheme=MXFP4_RCEIL`，`fp_layers=lm_head,mlp.gate,self_attn`，**iters=200**（开 FP8 KV/attn 时强制 iters=0），`disable_opt_rtn=True, low_gpu_mem_usage=True`，导出 llm_compressor。
- **DeepSeek（MLA）**：`scheme=MXFP4_RCEIL`，`fp_layers=lm_head,self_attn`，**iters=0**。
- **Dense（Qwen3-32B/8B/0.6B）**：`scheme=MXFP4`，`fp_layers=lm_head,self_attn`，iters=200。
- **Llama-4-Scout（多模态 MoE）**：`--fp_layers lm-head,self_attn,router,vision_model,multi_modal_projector,shared_expert`，scheme MXFP4，iters=0。

## 3. 共性规律与技巧

### 3.1 数值格式
- 全部 bits=4、group_size=32、对称、E8M0 scale；激活侧 MXFP4 动态量化。
- **MoE 多用 `MXFP4_RCEIL`**（rceil scale 取整更利于 vLLM fused-moe kernel）；dense 小模型常用普通 `MXFP4`。

### 3.2 Dense vs MoE 的配置差异（核心结论）
| | Dense | MoE |
|---|---|---|
| 量化主体 | 所有 Linear（或除 attn 外） | **routed experts**（占参数大头，最耐量化） |
| 必保 FP16 | lm_head、embedding | lm_head、embedding、**router/gate**、shared_experts（推荐） |
| attention | Intel 官方 recipe 整层 self_attn 保 16bit（更稳）；激进方案也量化 attn | MLA 模型（DeepSeek/Kimi）attn 必须保 FP16；普通 GQA MoE 可量化 attn |
| scheme | MXFP4 | MXFP4_RCEIL |
| iters | 200（小模型可 50/0） | 大旗舰普遍 iters=0 纯 RTN（精度损失 <1%）；追求极限用 200 |

### 3.3 忽略层频率排序
lm_head（100%）> embedding > **router/gate（MoE 100%）** > self_attn（MLA 模型必保，其余可选）> shared_experts > vision_tower/mm_projector（多模态）> layernorm（导出时自动排除）。

**原因**：attention 和 router 对 FP4 最敏感，expert 层最耐量化——这正是 MXFP4 特别适合 MoE 的原因。

### 3.4 实用技巧
1. **调优强度与模型规模反相关**：超大 MoE 用 RTN（iters=0）即可保持 99.3–99.8% 精度；小 dense 模型建议 iters=200 甚至 1000。
2. **混合精度**：GLM-5.2/Hy3/DSV4 等新做法是 experts FP4 + 敏感层 FP8（`mixed-precision` 导出），精度更好。
3. **enable_alg_ext=True** 对 MXFP4 有可见精度提升；`rotation_config="quarot"` 也可尝试。
4. **低显存**：`low_gpu_mem_usage=True` 省 ~20G；`gradient_accumulate_steps=8 + batch_size=1`。
5. **导出**：MXFP4 一律 `format="llm_compressor"`；推理用 vLLM + H100/B200（或 Triton≥3.4）。
6. **验证**：量化后用 lm-eval-harness 对比 BF16 baseline，MXFP4 RTN 正常应 ≥99% 精度保持。

## 4. 通用量化脚本

见同目录 `auto_round_quantize.py`，支持：
- 自动探测 dense / MoE（读 config 的 `num_experts_per_tok`/`n_routed_experts`/`num_local_experts` 等字段）；
- MoE 自动扫描 router/gate 层名（`.gate`、`router`、`gate_proj` 等后缀）并加入 FP16 保护；
- 内置 `dense` / `moe` / `moe-aggressive` 三档 ignore 预设，可用 `--extra_ignore_layers` / `--layer_config_json` 覆盖；
- 一键 MXFP4/MXFP4_RCEIL/W4A16，llm_compressor 导出。
