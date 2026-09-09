# Owner 任务书：qwen3-8-flash-next（MXFP8/MXFP4-Mixed × V1/V2/V3，day-0）

> 公共协议先读 `owner_tasks/README.md`；本文件只写该模型的差异。**这是 day-0 架构**（qwen4_exp），先读 aqa 仓库 `docs/day0_quantization_recipe.md`。

## 模型

- HF id：`Qwen/Qwen3.8-Flash-Next`（bf16 ~360GB；125B 总参 + 51B PLE + 4B MTP）
- 现有页面：`other-model-recipes/qwen3-8-flash-next/mxfp4-recipe.md`（RTN mixed）+ `accuracy-result.md`；mxfp8 页需新建
- HF 先例：`Qwen3.8-Flash-Next-MXFP4-Mixed-CT-AutoRound`
- 环境：**官方 vllm 镜像** `vllm/vllm-openai:qwen38-flash-next`（`env.type: docker`，`env.packages` 不要 vllm；量化用本地补丁版 auto-round wheel 见 day-0 文档）

## 变体配置（6 个 aqa run）

- **mxfp4-mixed**（已有 recipe 页）：ignore 沿用页面
  `visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection`；
  layer_config `{mlp.experts: {bits: 4, data_type: mx_fp}}`
- **mxfp8**（无页面，新建）：**禁止照抄 mixed 清单**——按 README 的 ignore 纪律
  结构分析 + 保守/激进两档探索后选定（该模型有 PLE/MTP/hyper_connection 等
  非常规模块，分析时逐一对照 modeling 源码确认角色后再定精度档）。

| 变体 | quant 段 | eval.model_args 追加 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method model_free, ignore=探索选定, format llm_compressor | — |
| mxfp8 v2 | autoround, iters 0, + static_kv_fp8 | `kv_cache_dtype: fp8` |
| mxfp8 v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |
| mxfp4-mixed v1 | scheme MXFP8 + layer_config, method model_free, ignore 沿用页面 | — |
| mxfp4-mixed v2 | autoround, iters 0, + static_kv_fp8（layer_config 保持） | `kv_cache_dtype: fp8` |
| mxfp4-mixed v3 | + static_attention_fp8 | `kv_cache_dtype: fp8` |

## 资源

- 量化：H200×4（docker 形态，官方镜像内跑）；评估 TP4
- PLE 51B 表注意：vllm 镜像外推理不可行，所有评估在镜像内

## 回写页面

- `mxfp4-recipe.md` 六段式补齐（Deployment notes 写明官方镜像依赖与原因）；**新建 `mxfp8-recipe.md`**
- `accuracy-result.md` 补 6 行 + evalscope 表
