# Owner 任务书：qwen-32b（MXFP8/MXFP4 × V1/V2/V3）

> 公共协议先读 `owner_tasks/README.md`；本文件只写该模型的差异。

## 模型

- HF id：先查 `overall-status.md` 的模型链接确认（页面名 qwen-32b，疑为 Qwen2.5-32B 系；**不要猜，以链接为准**）
- dense 32B（bf16 ~64GB）
- 现有页面：`key-model-recipes/qwen-32b/mxfp8-recipe.md`（RTN，exclude `lm-head`）+ `accuracy-result.md`；**mxfp4 页不存在，需新建**（front matter 照 mxfp8 页改 title/nav_order/permalink）

## 变体配置（6 个 aqa run）

- **mxfp8**（已有 recipe 页）：ignore 沿用页面 `lm_head`
- **mxfp4**（无页面，新建）：按 README 的 ignore 纪律做结构分析 + 保守/激进
  两档探索后选定（dense 模型起步参考：保守档加 `self_attn`，激进档仅
  `lm_head,embed_tokens`）。

| 变体 | quant 段 | eval.model_args 追加 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method model_free, ignore_layers "lm_head", format llm_compressor | — |
| mxfp8 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp8 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v1 | scheme MXFP4, method model_free, ignore=探索选定 | — |
| mxfp4 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |

## 资源

- 量化/评估均单卡 H200（num_gpus 1）

## 回写页面

- `mxfp8-recipe.md` 按六段式补齐；**新建 `mxfp4-recipe.md`**（nav_order 跟在 mxfp8 后）
- `accuracy-result.md`：补 mxfp8 v1/v2/v3 + mxfp4 v1/v2/v3 行
