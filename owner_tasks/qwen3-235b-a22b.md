# Owner 任务书：qwen3-235b-a22b（MXFP8/MXFP4 × V1/V2/V3）

> 公共协议先读 `owner_tasks/README.md`；本文件只写该模型的差异。

## 模型

- HF id：`Qwen/Qwen3-235B-A22B-Instruct-2507`（执行前用 `overall-status.md` 的链接二次确认）
- 235B 总参 / 22B 激活 MoE；bf16 ~470GB
- recipe 站页面（已存在，本次扩展）：`key-model-recipes/qwen3-235b-a22b/{mxfp4,mxfp8,nvfp4}-recipe.md` + `accuracy-result.md`（模板基准样例就在这个目录，照着写）
- 现有 recipe 要点：mxfp4 页 algo=Auto-round（**v1 用 tuning iters=200**），mxfp8 页 algo=RTN（**v1 用 model_free**）；ignore 均含 `lm_head,mlp.gate`

## 变体配置（6 个 aqa run）

| 变体 | quant 段 | eval.model_args 追加 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method model_free, ignore_layers "lm_head,mlp.gate", format llm_compressor | — |
| mxfp8 v2 | scheme MXFP8, method autoround, iters 0, static_kv_fp8, 同上 ignore | `kv_cache_dtype: fp8` |
| mxfp8 v3 | v2 + static_attention_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v1 | scheme MXFP4, method autoround, iters 200, ignore "lm_head,mlp.gate", format llm_compressor | — |
| mxfp4 v2 | mxfp4 校准路径（iters 0）+ static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v3 | v2 + static_attention_fp8 | `kv_cache_dtype: fp8` |

## 资源

- 量化：tuning 档 H200×4（num_gpus 4）；其余档 H200×2
- 评估：TP4（H200×4）；evalscope harness 会自起 vllm serve（同 TP）
- 预期时长：tuning 档小时级，其余档数十分钟；评估每档 lm_eval 4 任务 + evalscope 2 任务×5 次

## 回写页面

- `mxfp8-recipe.md` / `mxfp4-recipe.md`：按 demo 六段式补齐（Quantization/Resources/Evaluation/Deployment notes）
- `accuracy-result.md`：lm_eval 表补 v1/v2/v3 行（旧行保留标注）；evalscope 表填 aime25/gpqa_diamond avg@5
- nvfp4 页不动
