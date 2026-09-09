# Owner 任务书：FLUX.2-dev（MXFP8 / MXFP4，diffusion）

> 公共协议先读 `owner_tasks/README.md`；diffusion 模型没有 v2/v3（KV cache 是 LLM 概念），每模型只做 mxfp8 v1 + mxfp4 v1 两个变体。

## 模型

- HF id：`black-forest-labs/FLUX.2-dev`（已核实）
- 旗舰文生图 diffusion（bf16 权重 ~110GB，注意磁盘与下载时长）
- **recipe 站无对应目录，需新增**：建议 `key-model-recipes/flux-2-dev/`（front matter 参照 `flux/` 既有约定，注意别和 FLUX.1 的 `flux/` 混）；导航更新 `key-model-recipes.md` + `overall-status.md` + `index.md` Revision History
- 评估基线参考（同事 bf16 实测 DrawBench）：CLIP 28.5441 / CLIP-IQA 0.7927 / ImageReward 1.0661

## 变体配置（2 个 aqa run）

| 变体 | quant 段 | 备注 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method autoround, iters 0（diffusion 不走 model_free） | — |
| mxfp4 v1 | scheme MXFP4, method autoround, iters 0 | — |

diffusion 纪律（照 aqa `examples/run_diffusion_*.yaml` 的既有姿势）：
- **先 `hf download black-forest-labs/FLUX.2-dev --local-dir <path>`**（**gated repo**，
  env.sh 的 HF_TOKEN 需已接受其条款），model 写本地路径
- `extra_args` 必带：`--disable_model_free`、`--dataset coco2014`、`--num_inference_steps 28`、
  `--guidance_scale 3.5`（dev 惯例）；`export_format: auto_round`
- **ignore_layers（实证，不猜）**：我们的 smoke run（W4A16）没带 ignore 跑通；
  本模型 DiT 无 `adaLN_modulation`（与 Z-Image 不同）。mxfp8 v1 与 mxfp4 v1 均先不带
  ignore；**mxfp4 档若精度异常，把 `modulation`（double/single stream 共 3 个 Linear，
  参考 klein 系结构）加入保守档重跑对比**——diffusion 的 ignore 探索同样遵守
  README 的纪律（结构分析 → 保守/激进 → 精度裁决）。

## 评估（eval 段）

- `backend: vllm_omni`，`harness: diffusion`，`baseline: true`（要 bf16 对照行）
- benchmarks: `[drawbench_metrics, geneval]`（三指标 CLIP/CLIP-IQA/ImageReward + GenEval）
- diffusion 段参数按模型惯例（步数/guidance/分辨率 1024×1024，seed 42）

## 资源

- 量化/评估：H200×1-2（dev 权重大，留足磁盘；量化时长模型级小时）
- 全精度 bf16 baseline 评估也要跑（baseline: true 自动含）

## 回写页面

- 新建模型目录 + 桩页 + `mxfp8-recipe.md` / `mxfp4-recipe.md`（六段式；diffusion 的 Resources 表只列量化/推理两行即可，无 v2/v3）
- `accuracy-result.md`：DrawBench 三指标表（bf16/参考行 + mxfp8/mxfp4 行 + 相对保持）+ GenEval 表
