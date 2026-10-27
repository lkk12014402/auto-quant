# Diffusion（文生图量化 + 评估）用法

> 内核：`aqa/eval/harness/diffusion.py`（vllm-omni 生成 + 打分）+
> `aqa/backends/vllm_omni.py`。判断层：`skills/quantize/SKILL.md` 的 diffusion 段。
> 源码级机制深挖：`docs/autoround_diffusion_reference.md`；设计与里程碑：
> `docs/diffusion_support_plan.md`。

## 最小配置

```yaml
model: /path/to/Model         # diffusion 量化只认本地目录（先 hf download）
quant:
  scheme: W4A16               # W4A16 / MXFP8 / MXFP4（vllm-omni 上游 main 均支持）
  method: autoround
  export_format: auto_round   # vllm-omni 部署必须；fake 不能重新加载
  iters: 0                    # RTN 起步（DiT 校准代价 = nsamples × num_inference_steps 全去噪）
  extra_args:
    - --disable_opt_rtn
    - --disable_model_free    # 必加：iters=0+disable_opt_rtn 会被路由到 model_free，diffusion 下空量化
    - --ignore_layers
    - adaLN_modulation        # vllm-omni 设计上全精度保留的小层
    - --dataset
    - coco2014
eval:
  benchmarks: [drawbench_metrics]   # 三指标（clip / clip_iqa / imagereward）
  backend: vllm_omni
  harness: diffusion
  baseline: true
  diffusion:
    num_inference_steps: 8    # 蒸馏模型惯例（Z-Image）；FLUX.2-dev 这类非蒸馏要 28
    guidance_scale: 0.0       # 蒸馏 0.0；非蒸馏 3.5
    height: 1024
    width: 1024
    seed: 42
gate:
  max_accuracy_drop: 2.0      # 作用于 clip 指标（x100 标尺）
```

## 评估 recipe 三选一

| recipe | 测什么 | 判分 |
|---|---|---|
| `drawbench_clip` | 图文对齐 | CLIP score（单值） |
| `drawbench_metrics` | 图文对齐 + 画质 + 人类偏好 | CLIP / CLIP-IQA / ImageReward 三列 |
| `geneval` | 对象级组合生成正确性 | Mask2Former 检测 + CLIP ViT-L/14（0..1，6 类任务均值） |
| `dpgbench` | 密集 prompt 跟随 | mPLUG VQA 判分 |

三指标的 CLIP-IQA 复用同一 CLIP 模型（反义词 prompt 对），ImageReward 跑在
隔离的 transformers-v4 子进程环境里（`aqa/eval/image_metrics.py` 顶部有
完整的版本组合记录与原因——它和 v5 不兼容）。

## 实证过的模型

Z-Image-Turbo（W4A16 RTN 几乎无损：CLIP 31.78→31.66 / CLIP-IQA 0.704→0.703 /
ImageReward 0.993→0.970，GenEval 0.7579）、FLUX.2-dev / klein-4B / klein-9B、
DeepSeek-OCR-2。示例 yaml 在 `examples/run_diffusion_*.yaml`。

已发布 checkpoint 的逐项复现（MXFP8，含精确环境与步骤、ImageReward tokenizer
缓存坑）见 `cookbooks/Z-Image-Turbo-MXFP8-RTN-AutoRound-reproduction.md`，
可直接运行的配置是 `examples/repro_zimage_mxfp8_{drawbench,geneval}.yaml`。

## 已知坑（fix_playbook 已收录）

- `--model_name` 必须本地目录；`--disable_model_free` 必加；
- 导出可能弄坏 text encoder 权重名（丢 `model.` 前缀）——aqa 的
  `repair_diffusion_export` 自动修；
- 同进程二次加载 Omni 需要 teardown 分布式组（harness 已处理）；
- SVDQuant→Nunchaku 仅 FLUX.1，FLUX.2 不命中其 adapter 检测。
