# Owner 任务书：qwen3-30b-a3b-instruct-2507（MXFP8/MXFP4 × V1/V2/V3）

> 公共协议先读 `owner_tasks/README.md`；本文件只写该模型的差异。

## 模型

- HF id：`Qwen/Qwen3-30B-A3B-Instruct-2507`（bf16 ~61GB，MoE 128 专家）
- 现有页面：`key-model-recipes/qwen3-30b-a3b-instruct-2507/{mxfp4,mxfp8}-recipe.md`（RTN，exclude `lm-head,mlp.gate`）+ `accuracy-result.md`（另有 accuracy-result-2.md 和 turboquant 页，不动）
- 注意：该模型在 aqa 仓库有现成的三档混合样例（`examples/local_venv/run_qwen3_30b_*.yaml`），**本任务不做 mixed**，按 recipe 站既有口径跑全 scheme

## 变体配置（6 个 aqa run）

| 变体 | quant 段 | eval.model_args 追加 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method model_free, ignore_layers "lm_head,mlp.gate", format llm_compressor | — |
| mxfp8 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp8 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v1 | scheme MXFP4, method model_free, 同上 ignore, format llm_compressor | — |
| mxfp4 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |

## 资源

- 量化/评估均单卡 H200（num_gpus 1）；Thinking 判定：Instruct-2507 模板无 enable_thinking → 单模式

## 回写页面

- `mxfp4-recipe.md` / `mxfp8-recipe.md` 按六段式补齐
- `accuracy-result.md`：lm_eval 表补 6 行；evalscope 表填 aime25/gpqa_diamond avg@5
