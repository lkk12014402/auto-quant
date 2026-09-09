# Owner 任务书：qwen3-6-35b-a3b（MXFP8/MXFP4-Mixed × V1/V2/V3）

> 公共协议先读 `owner_tasks/README.md`；本文件只写该模型的差异。

## 模型

- HF id：`Qwen/Qwen3.6-35B-A3B`（以 overall-status.md 链接二次确认）
- MoE（bf16 ~70GB）；含 visual + linear_attn
- 现有页面：`other-model-recipes/qwen3-6-35b-a3b/mxfp4-recipe.md`（RTN mixed，exclude `visual,lm_head,mlp.gate,embed_tokens,self_attn,linear_attn`——attention 全保）+ `accuracy-result.md`；mxfp8 页需新建
- HF 先例：`Qwen3.6-35B-A3B-MXFP4-Mixed-CT-AutoRound`

## 变体配置（6 个 aqa run）

- **mxfp4-mixed**（已有 recipe 页）：ignore 沿用页面
  `visual,lm_head,mlp.gate,embed_tokens,self_attn,linear_attn`；
  layer_config `{mlp.experts: {bits: 4, data_type: mx_fp}}`
- **mxfp8**（无页面，新建）：**禁止照抄 mixed 清单**——按 README 的 ignore 纪律
  结构分析 + 保守/激进两档探索后选定（该模型 attention 敏感，激进档也建议
  保留 `self_attn,linear_attn`）。

| 变体 | quant 段 | eval.model_args 追加 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method model_free, ignore=探索选定, format llm_compressor | — |
| mxfp8 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp8 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |
| mxfp4-mixed v1 | scheme MXFP8 + layer_config, method model_free, ignore 沿用页面 | — |
| mxfp4-mixed v2 | autoround, iters 0, + static_kv_fp8（layer_config 保持） | `kv_cache_dtype: fp8` |
| mxfp4-mixed v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |

注：该模型 recipe 把 self_attn/linear_attn 全保 BF16——mxfp8 主体实际只覆盖
dense Linear，属正常设计，如实记录。

## 资源

- 量化 H200×2；评估 TP1-2 视显存

## 回写页面

- `mxfp4-recipe.md` 六段式补齐；**新建 `mxfp8-recipe.md`**
- `accuracy-result.md` 补 6 行 + evalscope 表
