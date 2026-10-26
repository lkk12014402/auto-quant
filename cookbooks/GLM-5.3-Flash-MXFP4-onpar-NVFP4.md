# GLM-5.3-Flash：与 NVIDIA NVFP4 模块级 on-par 的 MXFP4 量化

日期：2026-10-25 · 主机 8×B200 · owner: kaokaolv
产物：`GLM-5.3-Flash-MXFP4-AutoRound`（compressed-tensors）
对标：`nvidia/GLM-5.3-Flash-NVFP4`（modelopt 0.47.0，recipe `nvfp4_experts_dense_mlp-kv_fp8_cast`）
前作参考：`intel-ai/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`

## 1. on-par 的定义（本模型）

NVIDIA 这个 checkpoint **没有 FP8 权重**——`hf_quant_config.json` 是
`quant_algo: NVFP4` + `kv_cache_quant_algo: FP8`；`.quant_summary.txt` 中启用的权重量化器
只有 **dense MLP（layers 0-2）** 与 **routed experts（layers 3-44）**。
所以映射是：

| NVIDIA | 我们 |
|---|---|
| NVFP4（dense MLP + routed experts，group 16） | **MXFP4 W4A4**（group 32，E8M0） |
| BF16（self_attn / router / shared_experts / layers.45 / visual / embed / lm_head） | **BF16** |
| FP8 **KV cache**（runtime） | 推理时 `--kv-cache-dtype fp8`（同为 runtime） |

注意与 intel-ai 前作的差异（本任务以 NVIDIA 为准）：前作把 **shared_experts 与 dense MLP
量化成 MXFP8**；这里 **shared_experts 保 BF16、dense MLP 用 MXFP4**。

## 2. 逐张量证据（safetensors header，HTTP Range）

| 模块 | NVIDIA dtype / shape | 判定 |
|---|---|---|
| `layers.0.mlp.down_proj.weight` | U8 [4096,6144] + `weight_scale` F8_E4M3 [4096,768]=in/16 | NVFP4 |
| `layers.3.mlp.experts.0.{gate,up,down}_proj.weight` | U8 + F8_E4M3 in/16 | NVFP4 |
| `layers.0.self_attn.q_proj.weight` | BF16 [8192,4096] | BF16 |
| `layers.3.mlp.shared_experts.gate_proj.weight` | BF16 [2048,4096] | BF16 |
| `layers.3.mlp.gate.weight` | BF16 [288,4096] | BF16 |
| `layers.45.mlp.experts.0.down_proj.weight` / `layers.45.eh_proj.weight` | BF16 | BF16（MTP 整层） |
| `model.visual.*` | `.quant_summary` 全 `disabled` | BF16 |

集合计数：NVFP4 = **36 297** = 36 288 routed experts + 9 dense MLP。

## 3. Recipe

```bash
auto-round --model_name zai-org/GLM-5.3-Flash-BF16 \
  --model_free --scheme MXFP4 \
  --ignore_layers "lm_head,embed_tokens,visual,indexer,weights_proj,self_attn,mlp.shared_experts,layers.45,eh_proj,enorm,hnorm,shared_head,norm,A_log,dt_bias,conv1d" \
  --layer_config '{"model.language_model.layers.0.mlp.gate_proj":{"bits":4,"data_type":"mx_fp"},"model.language_model.layers.0.mlp.up_proj":{"bits":4,"data_type":"mx_fp"},"model.language_model.layers.0.mlp.down_proj":{"bits":4,"data_type":"mx_fp"}}' \
  --format llm_compressor \
  --output_dir GLM-5.3-Flash-MXFP4-AutoRound
```

**范围预演（不加载权重，纯名字匹配，BF16 index 38 230 个 `.weight`）**：

```
quantized 36 297   ignored 1 786   hard-skip 147
NVIDIA NVFP4 modules 36 297 | missing 0 | extra 0
```

## 4. 必踩的坑：auto-round 对 `glm5_next` 的 predefined ignore

`special_model_handler.py` 注册了 `indexer, layers.0.mlp, self_attn, weights_proj`，
与 `--ignore_layers` **取并集、无开关**。其中 `layers.0.mlp` 来自
`get_glm_flash_ignore_layers` 读 `model.config.first_k_dense_replace`——model_free 的伪
config 是 `SimpleNamespace(**config.json)`，该字段在 `text_config` 里 → 回落
`num_dense_layer=1` → **只保层 0，层 1/2 照常量化**，正好把 NVIDIA 量化了的层 0 dense MLP 漏掉。

绕法（不改库）：`should_ignore` 对 **layer_config 精确全名**优先返回 False
（`model_free_utils.py:513-524`），把层 0 的 3 个投影写进 `--layer_config` 即可。
子串 key（如 `mlp.experts`）绕不过，必须全名。

## 5. 评估

lm_eval 0.4.13 + vLLM（`.venv-vllm-main`，nightly `46d2b23ac`，registry 含
`Glm5NextForConditionalGeneration`），TP=2、`kv_cache_dtype=fp8`、`block_size=256`、
`max_model_len=8192`、`enforce_eager`，4 任务全量：
piqa / hellaswag / mmlu（0-shot）+ gsm8k（5-shot，仓库自带 `chat_template.jinja`）。
baseline = 同一 BF16 checkpoint 同机同口径。

结果（`runs/eval_quant_full`，4 任务全量，1838/10042/14042/1319 样本）：

| Benchmark | MXFP4-Mixed (full) |
|---|---:|
| piqa | 82.75 |
| hellaswag | 68.25 |
| mmlu | 85.88 |
| gsm8k | 97.50 |
| **Average** | **83.60** |

> baseline full 尚未完成：642.65 GB BF16 在 2×B200 上装不下，`eval_baseline_full.yaml`
> 已改为 6 卡；等 GPU4-7 释放后再跑，之后才能给 on-par 结论。

## 7. 必踩的坑：auto-round 导出 config 不能直接被 vLLM 加载

vLLM 的 GLM-5.3 模型类用 `language_model.model.*` 命名，而 auto-round 导出的是
`model.language_model.*`；并且该模型类没有 `packed_modules_mapping`，fused
`gate_up_proj` 无法用 `gate_proj` 的 ignore 命中。结果就是 vLLM 把 BF16 模块当
MXFP4 加载，报 KeyError / 权重 shape 不匹配。

**确定性修法**（已落地为 `aqa.quantize.export_repair --kind glm5_vllm`）：

```bash
python -m aqa.quantize.export_repair <quantized_dir> --kind glm5_vllm
```

它把 `config_groups.group_0.targets` 从 `["Linear"]` 改成从 index 反查的显式
vLLM 名字（dense MLP 层 0-2 的 `gate_up_proj`/`down_proj` + `"RoutedExperts"`），
`ignore` 设为 `language_model.model.layers.45.mlp.experts`，并把原始 config 备份为
`.pre_export_repair_config.json`。本次 run 在 19:36 手工执行过同样的 rewrite，
已记入 `owner_actions.jsonl`；新 run 应在 YAML 里设 `quant.export_repair: glm5_vllm`。

## 6. 部署注意

- `--kv-cache-dtype fp8`（对齐 NVIDIA 的 FP8 KV）。
- 本机评估需 `NCCL_NVLS_ENABLE=0`（NVSwitch/Fabric Manager NVLS multicast 初始化失败）。
- TP 需要能整除注意力头数（64）。
