# Owner 任务书：FLUX.2-klein-4B（MXFP8 / MXFP4，diffusion）

> 公共协议先读 `owner_tasks/README.md`；diffusion 无 v2/v3，只做 mxfp8 v1 + mxfp4 v1。

## 模型

- HF id：`black-forest-labs/FLUX.2-klein-4B`（已核实，注意大写 B）
- 轻量文生图（bf16 ~8GB 量级）
- **recipe 站无对应目录，需新增**：`other-model-recipes/flux-2-klein-4b/`（导航三处同步：栏目 md + overall-status + index.md 修订表）
- 评估基线参考（同事 bf16 实测 DrawBench）：CLIP 28.1660 / CLIP-IQA 0.8370 / ImageReward 1.0380

## 变体配置（2 个 aqa run）

| 变体 | quant 段 |
|---|---|
| mxfp8 v1 | scheme MXFP8, method autoround, iters 0 |
| mxfp4 v1 | scheme MXFP4, method autoround, iters 0 |

diffusion 纪律同 flux2-dev 任务书（本地目录加载、disable_model_free、coco2014、
export_format auto_round；本模型 8 步 / guidance 0.0）。

**ignore_layers（实证）**：我们 smoke run（MXFP8）没带 ignore 跑通。DiT 里只有
3 个 modulation Linear（`double_stream_modulation_img/txt`、`single_stream_modulation`），
**没有** `adaLN_modulation`。mxfp8 v1 / mxfp4 v1 均先不带 ignore；mxfp4 若掉点，
保守档加 `modulation` 重跑对比（见 README ignore 纪律）。

## 评估

- `backend: vllm_omni`，`harness: diffusion`，`baseline: true`
- benchmarks: `[drawbench_metrics, geneval]`；推理参数按 klein 惯例（步数/guidance 以模型卡为准，写入 journal）

## 资源

- H200×1 即可（量化与评估）

## 回写页面

- 新建 `other-model-recipes/flux-2-klein-4b/`（桩页 + mxfp8/mxfp4 两 recipe 页 + accuracy-result.md）
