# Owner 任务书：qwen3-5-9b（MXFP8/MXFP4 × V1/V2/V3）

> 公共协议先读 `owner_tasks/README.md`；本文件只写该模型的差异。

## 模型

- HF id：`Qwen/Qwen3.5-9B`（以 overall-status.md 链接二次确认）
- 现有页面：`other-model-recipes/qwen3-5-9b/mxfp8-recipe.md`（RTN，exclude `lm_head,embed_tokens,visual,in_proj_a,in_proj_b,conv1d,gate`——含 visual，是 omni 形态）+ `accuracy-result.md`；mxfp4 页需新建

## 变体配置（6 个 aqa run）

- **mxfp8**（已有 recipe 页）：ignore 沿用页面 `lm_head,embed_tokens,visual,in_proj_a,in_proj_b,conv1d,gate`
- **mxfp4**（无页面，新建）：**禁止照抄 mxfp8 的清单**——先做结构分析出逐模块
  精度表初稿，再跑保守/激进两档 ignore 探索（见 README 的 ignore 纪律），
  由精度对比选定入页清单。

| 变体 | quant 段 | eval.model_args 追加 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method model_free, ignore 沿用页面, format llm_compressor | — |
| mxfp8 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp8 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v1 | scheme MXFP4, method model_free, ignore=探索选定 | — |
| mxfp4 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |

探索档参考（owner 可按结构分析修正，跑 sweep 自定义候选）：
保守档 ignore 在页面清单上加 `self_attn`；激进档只保 `lm_head,embed_tokens,mlp.gate`。

## 资源

- 单卡 H200（num_gpus 1）；含 visual 的模型评估若报视觉相关错，按 fix-loop 协议处理并记录

## 回写页面

- `mxfp8-recipe.md` 六段式补齐；**新建 `mxfp4-recipe.md`**
- `accuracy-result.md` 补 6 行 + evalscope 表
