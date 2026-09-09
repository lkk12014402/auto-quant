# Owner 任务书：Z-Image-Turbo（MXFP8 / MXFP4，diffusion）

> 公共协议先读 `owner_tasks/README.md`；diffusion 无 v2/v3，只做 mxfp8 v1 + mxfp4 v1。

## 模型

- HF id：`Tongyi-MAI/Z-Image-Turbo`（已核实）
- 6B S3-DiT 蒸馏文生图（8 步、不做 CFG——guidance 0.0）
- **recipe 站无对应目录，需新增**：`other-model-recipes/z-image-turbo/`（导航三处同步）
- 评估基线参考（同事 bf16 实测 DrawBench）：CLIP 27.7382 / CLIP-IQA 0.8459 / ImageReward 0.9501；GenEval 官方口径 ~0.84（对齐验证参考）

## 变体配置（2 个 aqa run）

| 变体 | quant 段 |
|---|---|
| mxfp8 v1 | scheme MXFP8, method autoround, iters 0 |
| mxfp4 v1 | scheme MXFP4, method autoround, iters 0 |

可直接参照 aqa 仓库现成样例改造：`examples/run_diffusion_zimage.yaml`（W4A16 版，改成目标 scheme）与 `examples/run_diffusion_zimage_geneval.yaml`。
diffusion 纪律同 flux2-dev 任务书（本地目录加载、disable_model_free、coco2014、
export_format auto_round、8 步 / guidance 0.0）。

**ignore_layers（实证）**：`--ignore_layers adaLN_modulation` 是**必须项**——
该模型 DiT 有 66 个 adaLN_modulation 张量（index.json 实测），且 vllm-omni 对
adaLN/t_embedder 按设计全精度保留（quant_config=None），量化它们会加载失败。
我们 zimage smoke 即按此跑通。

## 评估

- `backend: vllm_omni`，`harness: diffusion`，`baseline: true`
- benchmarks: `[drawbench_metrics, geneval]`；`num_inference_steps: 8`，`guidance_scale: 0.0`，1024×1024，seed 42

## 资源

- H200×1

## 回写页面

- 新建 `other-model-recipes/z-image-turbo/`（桩页 + mxfp8/mxfp4 两 recipe 页 + accuracy-result.md：DrawBench 三指标表 + GenEval 表，GenEval 行注明与官方 0.84 的对齐情况）
