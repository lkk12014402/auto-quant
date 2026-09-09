# Owner 任务书：qwen3-5-35b-a3b（MXFP8/MXFP4 × V1/V2/V3）

> 公共协议先读 `owner_tasks/README.md`；本文件只写该模型的差异。

## 模型

- HF id：`Qwen/Qwen3.5-35B-A3B`（以 overall-status.md 链接二次确认）
- MoE（bf16 ~70GB）；含 visual/linear_attn 组件（in_proj_a/b）
- 现有页面：`other-model-recipes/qwen3-5-35b-a3b/mxfp8-recipe.md`（RTN，exclude `lm_head,embed_tokens,visual,in_proj_a,in_proj_b`）+ `accuracy-result.md`；mxfp4 页需新建

## 变体配置（6 个 aqa run）

- **mxfp8**（已有 recipe 页）：ignore 沿用页面 `lm_head,embed_tokens,visual,in_proj_a,in_proj_b`
- **mxfp4**（无页面，新建）：**禁止照抄 mxfp8 清单**——按 README 的 ignore 纪律：
  结构分析出逐模块精度表初稿 → 保守/激进两档探索 → 精度选定。

| 变体 | quant 段 | eval.model_args 追加 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method model_free, ignore 沿用页面, format llm_compressor | — |
| mxfp8 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp8 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v1 | scheme MXFP4, method model_free, ignore=探索选定 | — |
| mxfp4 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |

## 资源

- 量化 H200×2（num_gpus 2）；评估 TP1-2 视显存

## 回写页面

- `mxfp8-recipe.md` 六段式补齐；**新建 `mxfp4-recipe.md`**
- `accuracy-result.md` 补 6 行 + evalscope 表
