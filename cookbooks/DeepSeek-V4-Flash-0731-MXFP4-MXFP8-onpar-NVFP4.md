# DeepSeek-V4-Flash-0731：与 NVIDIA NVFP4 模块级 on-par 的 MXFP4(experts) + MXFP8(其余) 量化

日期：2026-10-25 · 主机：8×B200 · owner: kaokaolv
产物：`DeepSeek-V4-Flash-0731-MXFP4-Mixed-CT-AutoRound`
前作对照：`intel-ai/DeepSeek-V4-Flash-0731-MXFP4-Mixed-CT-AutoRound`
对标：`nvidia/DeepSeek-V4-Flash-0731-NVFP4`

## 1. 目标与结论

「与 NVIDIA NVFP4 on-par」在本次任务里的定义是**量化模块集合一致**，只做 scheme 映射：

| NVIDIA 里的 scheme | 我们的 scheme |
|---|---|
| NVFP4（routed experts, group 16） | **MXFP4**（E2M1, group 32, E8M0） |
| 原生 FP8 E4M3（128×128 block, ue8m0） | **MXFP8**（E4M3, group 32, E8M0） |
| 原生 MXFP4（MTP experts, group 32） | **MXFP4**（不变） |
| BF16/F32 | **BF16/F32**（不量化） |

## 2. 为什么模块集合是这样（逐张量证据）

NVIDIA 模型的 `quantization_config`：

```json
{"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0",
 "weight_block_size": [128,128],
 "quant_algo": "MIXED_PRECISION", "moe_quant_algo": "NVFP4", "group_size": 16,
 "ignore": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"],
 "quantized_layers": {"layers.{0..42}.ffn.experts": {"quant_algo": "NVFP4"}}}
```

- `quantized_layers` **只有** routed experts（NVFP4）；
- `quant_method: fp8` + 128×128 block 是**继承自源 checkpoint 的原生 DeepSeek FP8**，
  即 attention / shared experts / indexer.wq_b / mtp linears 仍是 FP8；
- `mtp.*` 被排除在 NVFP4 之外，保持源格式（attention FP8、experts 原生 MXFP4）；
- model card 原文：*"Only the routed MoE experts are quantized. DSpark speculative-decoding
  heads are carried through unquantized and unmodified."*

safetensors header 实测（HTTP Range 读头，不下载权重）：

| 模块 | NVIDIA dtype / shape | 判定 |
|---|---|---|
| `layers.*.ffn.experts.*.w{1,2,3}.weight` | U8 `[2048,2048]` + `weight_scale` F8_E4M3 `[out,in/16]` | NVFP4 packed |
| `attn.{wq_a,wq_b,wkv,wo_a,wo_b}` / `ffn.shared_experts.*` / `indexer.wq_b` | F8_E4M3 + `.scale` F8_E8M0 `[out/128,in/128]` | 原生 FP8 |
| `mtp.*.ffn.experts.*` | I8 `[2048,2048]` + `.scale` E8M0 `[out,in/32]` | 原生 MXFP4 |
| `mtp.*.attn.*` / `mtp.*.ffn.shared_experts.*` / `mtp.0.main_proj` | F8_E4M3 + E8M0 block | 原生 FP8 |
| compressor / indexer.compressor / indexer.weights_proj / router / hc_* / embed+head / markov+confidence heads | BF16 / F32 | 不量化 |

计数：NVFP4 33 024；原生 FP8 390；原生 MXFP4（MTP experts）2 304。

## 3. 量化 recipe（auto-round model_free）

```bash
auto-round --model_name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --model_free --scheme MXFP8 \
  --ignore_layers compressor,indexer.weights_proj,markov_head,confidence_head,ffn.gate,head \
  --layer_config '{"ffn.experts":{"bits":4,"data_type":"mx_fp"}}' \
  --format llm_compressor \
  --output_dir <out>
```

- `--scheme MXFP8` → 所有未被覆盖的 Linear：MXFP8 W8A8（group 32，E8M0）。
- `--layer_config {"ffn.experts": …}` → 子串匹配 `layers.*.ffn.experts` 与
  `mtp.*.ffn.experts`，压到 MXFP4 W4A4。
- `--ignore_layers` 对齐 NVIDIA 的不量化集合；**该模型头部叫 `head` 而非 `lm_head`**，
  router 叫 `ffn.gate`，MTP 的 `markov_head`/`confidence_head` 必须显式排除。
- `model_free`（免校准）：auto-round ≥0.17 的 `deepseek_v4` 源张量归一化
  （`preprocess_model_type_source_tensors`）把源 FP8 的 128×128 block scale 展开为
  per-32 E8M0，`_handle_mxfp_source_tensors` 对同 scheme 层做 **bit-exact passthrough**
  （专家 MXFP4→MXFP4 权重零改动；FP8→MXFP8 只是 scale 布局展开）。因此本产物
  不是"重新量化"，而是**格式重编码**，与 NVIDIA 的 `--cast_mxfp4_to_nvfp4` 思路同源。

## 4. 评估口径

lm_eval 0.4.13 + vLLM，TP=2，`kv_cache_dtype=fp8`、`block_size=256`、
`max_model_len=8192`、`enforce_eager`，4 任务全量：
piqa / hellaswag / mmlu（0-shot loglikelihood）+ gsm8k（5-shot generative，chat template）。

官方 checkpoint（原生 FP4+FP8）与量化产物同口径各跑一遍；DeepSeek 未发布 BF16 版，
官方 checkpoint 就是最强可用 reference。

结果（lm_eval 0.4.13 + vLLM TP=2，全量样本；gsm8k 5-shot strict-match）：

| task | 官方 checkpoint（同机基线） | 本产物 MXFP4-Mixed | intel-ai 前作（参考） |
|---|---:|---:|---:|
| gsm8k | 93.56 | 0.9439 | 0.9568 |
| mmlu | 86.31 | 0.8643 | 0.8631 |
| piqa | 82.64 | 0.8210 | 0.8232 |
| hellaswag | 68.02 | 0.6784 | 0.6788 |
| **mean** | **82.63** | **0.8269** | **0.8305** |

（官方 checkpoint 基线本次同机同口径重跑（2026-10-25，TP=2，全量样本）；前作列为 `intel-ai/DeepSeek-V4-Flash-0731-MXFP4-Mixed-CT-AutoRound` 的公开数字。）

## 5. 服务注意事项

- DSv4 的 fp8_ds_mla KV 布局只接受 `--kv-cache-dtype fp8`（bf16/auto 仅 FlashInfer
  sparse 后端可用）。
- **compressed-tensors MXFP8 wo_a 需要 vLLM patch**：上游 `deepseek_v4` 的
  `o_proj` 只认原生 128×128 block FP8；CT 的 per-32 scale 会进 `fp8_einsum` 报错。
  见 `patches/evaluate/vllm_dsv4_ct_mxfp8_bmm_wo_a.md`（两处最小改动，接上
  vLLM 自带的 `DeepGemmMxfp8BmmLinearKernel`）。
- TP 必须整除 64 个 attention head。

## 6. 可复现性

- auto-round：`intel/auto-round` @ `80b5f87`（2026-10-25），editable 安装。
- vLLM：nightly wheel commit `46d2b23ac`（`.venv-vllm-main`）+ 上述 patch。
- run yaml：`examples/local_venv/run_dsv4_flash_0731_mxfp4_mxfp8.yaml`。

## 附：本产物的实测验收（2026-10-25）

- 产物：167.08 GB / 48 shards / 72 317 tensors（源 166.88 GB，+0.2 GB 为 per-32 scale 开销）。
- `audit_export.py`（只读 safetensors header）PASS：
  - `layers.*.ffn.experts.*.w{1,2,3}.weight_packed` U8 × 33 024 + `weight_scale` U8（per-32）
  - `mtp.*.ffn.experts.*` U8 × 2 304，同上
  - attn 230 / shared_experts 138 / indexer.wq_b 21 / mtp.main_proj 1 = 390 个 `F8_E4M3 + U8 weight_scale`
  - compressor 124 / indexer.weights_proj 21 / router 46 / markov+confidence 3 / embed+head 2 = BF16
  - 43/43 主层 + mtp.0-2 齐全；`quantization_config.format=mixed-precision`，ignore 196 项
- 与 NVIDIA 计数逐项一致：NVFP4 33 024 → MXFP4 33 024；FP8 390 → MXFP8 390；
  MTP 原生 MXFP4 2 304 → MXFP4 2 304。

## 附：本机评估环境坑

- **NCCL NVLS**：本机 NVSwitch/Fabric Manager 状态下 `ncclCommInitRank` 报
  `Failed to bind NVLink SHARP (NVLS) Multicast memory ... CUDA error 401`。
  评估命令必须带 `NCCL_NVLS_ENABLE=0`（最小复现：2 rank torch all_reduce）。
- **lm-eval 0.4.13**：`--model_args` 先走 `try_parse_json`，嵌套 dict（逗号）会被切碎，
  不要在 `eval.lm_eval.model_args` 里放 `kernel_config` 这类嵌套值。
