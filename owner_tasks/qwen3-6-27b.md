# Owner 任务书：qwen3-6-27b（MXFP8/MXFP4-HMT × V1/V2/V3）

> 公共协议先读 `owner_tasks/README.md`；本文件只写该模型的差异。

## 模型

- HF id：`Qwen/Qwen3.6-27B`（以 overall-status.md 链接二次确认）
- dense 27B（bf16 ~54GB）
- 现有页面：`other-model-recipes/qwen3-6-27b/mxfp4-recipe.md`（**ITERS=200 + HMT（hadamard）**，exclude `lm_head,embed_tokens`）+ `accuracy-result.md`；mxfp8 页需新建
- HF 先例：`Qwen3.6-27B-MXFP4-HMT-AutoRound`

## 变体配置（6 个 aqa run）

- **mxfp4**（已有 recipe 页，ITERS=200 + HMT）：ignore 沿用页面 `lm_head,embed_tokens`
- **mxfp8**（无页面，新建）：**禁止照抄 mxfp4 清单**——按 README 的 ignore 纪律做
  结构分析 + 保守/激进两档探索后选定（dense 模型起步参考：保守档加 `self_attn`，
  激进档仅 `lm_head,embed_tokens`）。

| 变体 | quant 段 | eval.model_args 追加 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method model_free, ignore=探索选定, format llm_compressor | — |
| mxfp8 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp8 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |
| mxfp4 v1 | scheme MXFP4, method autoround, iters 200, algorithm: hadamard,auto_round, ignore 沿用页面 | — |
| mxfp4 v2 | 校准路径 iters 0 + static_kv_fp8（保留 hadamard 链） | `kv_cache_dtype: fp8` |
| mxfp4 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |

mxfp4 v1 对应 aqa yaml 字段：`algorithm: "hadamard,auto_round"`（HMT）；若校准路径带 hadamard 失败，记录现象并按 fix-loop 协议处理（不要静默改配置）。

## 资源

- 单卡 H200（量化 tuning 档 num_gpus 1 即可，HMT 稍慢）

## 回写页面

- `mxfp4-recipe.md` 六段式补齐（HMT 要点写进 Deployment notes）；**新建 `mxfp8-recipe.md`**
- `accuracy-result.md` 补 6 行 + evalscope 表
