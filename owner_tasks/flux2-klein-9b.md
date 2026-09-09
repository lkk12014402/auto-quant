# Owner 任务书：FLUX.2-klein-9B（MXFP8 / MXFP4，diffusion）

> 公共协议先读 `owner_tasks/README.md`；diffusion 无 v2/v3，只做 mxfp8 v1 + mxfp4 v1。

## 模型

- HF id：`black-forest-labs/FLUX.2-klein-9B`（已核实；注意另有 `FLUX.2-klein-base-9B` 和 `-fp8` 变体，**不要下错**）
- 轻量文生图（bf16 ~18GB 量级）
- **recipe 站无对应目录，需新增**：`other-model-recipes/flux-2-klein-9b/`（导航三处同步）
- 评估基线参考（同事 bf16 实测 DrawBench）：CLIP 29.0622 / CLIP-IQA 0.8559 / ImageReward 1.1889

## 变体配置（2 个 aqa run）

| 变体 | quant 段 |
|---|---|
| mxfp8 v1 | scheme MXFP8, method autoround, iters 0 |
| mxfp4 v1 | scheme MXFP4, method autoround, iters 0 |

diffusion 纪律同 flux2-dev 任务书（本地目录加载、disable_model_free、coco2014、
export_format auto_round；本模型 8 步 / guidance 0.0）。

**ignore_layers（实证）**：smoke run（W4A16）没带 ignore 跑通；结构同 klein-4B
（3 个 modulation Linear，无 adaLN_modulation）。mxfp8 v1 / mxfp4 v1 先不带 ignore；
mxfp4 掉点则保守档加 `modulation` 重跑对比。

## 评估

- `backend: vllm_omni`，`harness: diffusion`，`baseline: true`
- benchmarks: `[drawbench_metrics, geneval]`；推理参数按 klein 惯例（模型卡为准）

## 资源

- H200×1-2

## 回写页面

- 新建 `other-model-recipes/flux-2-klein-9b/`（桩页 + mxfp8/mxfp4 两 recipe 页 + accuracy-result.md）
