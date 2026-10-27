# Z-Image-Turbo MXFP8（INCModel3 发布版）复现

- 日期：2026-10-25
- 目标 card：[INCModel3/Z-Image-Turbo-MXFP8-RTN-AutoRound](https://huggingface.co/INCModel3/Z-Image-Turbo-MXFP8-RTN-AutoRound)
- 底模：[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)（6B S3-DiT，蒸馏文生图）
- 复现方式：**直接评测已发布 checkpoint**（14.7 GB），不重新量化
- 结论：四项指标全部复现，DrawBench 三项差异 ≤0.2%，GenEval 差 0.65pp

## 0. 结论速览

| 指标 | Card (MXFP8) | 复测 | 差异 |
|---|---:|---:|---:|
| DrawBench CLIP | 31.79 | **31.774** | -0.016 |
| DrawBench CLIP-IQA | 70.41 | **70.577** | +0.167 |
| DrawBench ImageReward | 0.97 | **0.983** | +0.013 |
| GenEval | 0.760 | **0.7535** | -0.0065 |

- DrawBench：200 prompts，生成口径 8 steps / guidance 0.0 / 1024×1024 / seed 42。
- GenEval：553 prompts × 4 图 = 2212 样本，Mask2Former + CLIP ViT-L/14 判分；0.7535 与 card 的 0.760 差 0.65pp，在采样噪声内（本机历史 bf16 0.7579，card bf16 列 0.757）。
- Card 的 BF16 列（31.73 / 70.57 / 1.00 / 0.757）与本机历史 bf16 测量一致（CLIP 31.7837、GenEval 0.7579 见 `aqa_results/zimage_full`、`aqa_results/zimage_geneval_full`；CLIP-IQA 70.3335、ImageReward 1.0094 见 `examples/run_zimage_turbo_mxfp8_evalonly.yaml` 注释）。本次未重跑 bf16。

## 1. 复现对象与权重

已发布 checkpoint 的结构（从 repo 直接读取，非猜测）：

| 项 | 值 |
|---|---|
| 体积 | 14.71 GB，26 个文件 |
| 结构 | `transformer/` 7 shards + `text_encoder/` 3 shards + `vae/` + tokenizer/scheduler |
| 量化配置 | `transformer/quantization_config.json`：MXFP8（`bits=8, act_bits=8, data_type=mx_fp, group_size=32`） |
| 量化方法 | `quant_method=auto-round`，`packing_format=auto_round:llm_compressor`，`autoround_version=0.15.0` |
| 量化范围 | `block_name_to_quantize=noise_refiner,context_refiner,layers` |
| 全精度保留 | `adaLN_modulation`（34 项 `extra_config`，bits=16），以及 text encoder / VAE |
| 附带文件 | `aqa_export_repair.json`（导出时修复 text_encoder 的记录） |

下载（走代理时先 `export http_proxy/https_proxy`）：

```bash
HF_HOME=/home/kaokaolv/hf_home_aqa \
  /home/kaokaolv/.venv-omini-latest/bin/hf download \
  INCModel3/Z-Image-Turbo-MXFP8-RTN-AutoRound \
  --local-dir /home/kaokaolv/models/Z-Image-Turbo-MXFP8-RTN-AutoRound
```

## 2. 运行环境

**主环境：`/home/kaokaolv/.venv-omini-latest`**（不要用 `.venv-vllm-omni`：它缺 `datasets`；`.venv-aqa-gpu` 的 vllm-omni 0.27 未验证 MXFP8）

| 包 | 版本 |
|---|---|
| vllm | 0.29.0 |
| vllm-omni | 0.29.0rc2.dev161+g1c7476ec1 |
| transformers | 5.14.1 |
| torch | 2.13.0+cu132 |
| diffusers | 0.40.0 |
| datasets | 5.0.1 |
| flashinfer-python | 0.6.18 |
| pillow | 12.3.0 |
| python | 3.12.3 |

- aqa：**没有安装进该 venv**，用 `PYTHONPATH=<repo>` + `python -m aqa.cli` 从仓库加载。
- 关键验证：`from vllm_omni.quantization.inc_config import IncMxfp8OfflineLinearMethod` 必须可导入；运行时日志应出现 `Building quantization config: auto-round` 与 `Using MarlinMxfp8LinearKernel for MXFP8 GEMM`。
- 硬件：单卡 NVIDIA B200（182 GB），模型加载后约 20 GB 显存。

**环境变量**

| 变量 | 值 | 作用 |
|---|---|---|
| `PYTHONPATH` | `/home/kaokaolv/aqa/latest/auto-quant` | 从仓库加载 aqa |
| `HF_HOME` | `/home/kaokaolv/hf_home_aqa` | 复用 DrawBench / Mask2Former / CLIP / bert 缓存（ImageReward 子进程也继承） |
| `HF_HUB_OFFLINE` | `1` | 缓存齐全时避免联网卡顿；首次准备缓存时不要设 |
| `VLLM_DISABLE_COMPILE_CACHE` | `1` | 改量化/换 checkpoint 后避免复用陈旧 inductor 图 |
| `CUDA_VISIBLE_DEVICES` | `0`（DrawBench）/ `1`（GenEval） | 两阶段分开跑 |

**缓存准备**（首次需要联网，之后可离线）：

```bash
export HF_HOME=/home/kaokaolv/hf_home_aqa
HF=/home/kaokaolv/.venv-omini-latest/bin/hf
$HF download sayakpaul/drawbench --repo-type dataset
$HF download facebook/mask2former-swin-small-coco-instance
$HF download openai/clip-vit-base-patch32
$HF download openai/clip-vit-large-patch14
# ImageReward 的 tokenizer：HF 别名在 CLI 里会 404，下 canonical repo 再建缓存别名
$HF download google-bert/bert-base-uncased
ln -sfn models--google-bert--bert-base-uncased $HF_HOME/hub/models--bert-base-uncased
```

**ImageReward 隔离环境**：`~/.cache/aqa/envs/imagereward`（transformers 4.44.2 / diffusers 0.29.2 / image-reward 1.5），由 `aqa/eval/image_metrics.py` 自动创建和调用；可用 `AQA_IMAGEREWARD_ENV` 覆盖路径。它跑在子进程里，torch 用主环境的（隔离目录里的 torch 生态会被清掉）。

## 3. 复现步骤

配置已固化在仓库：

- `examples/repro_zimage_mxfp8_drawbench.yaml`
- `examples/repro_zimage_mxfp8_geneval.yaml`

### 3.1 DrawBench（CLIP / CLIP-IQA / ImageReward）

```bash
cd /home/kaokaolv/aqa/latest/auto-quant
export PYTHONPATH=$PWD
export HF_HOME=/home/kaokaolv/hf_home_aqa
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export VLLM_DISABLE_COMPILE_CACHE=1

/home/kaokaolv/.venv-omini-latest/bin/python -m aqa.cli run \
  examples/repro_zimage_mxfp8_drawbench.yaml
```

预期日志顺序：`Building quantization config: auto-round` → `Using MarlinMxfp8LinearKernel for MXFP8 GEMM` → 200 张图（约 1.5–3 s/张）→ 三项打分。结果在
`<output.dir>/eval/accuracy.json`，键为 `drawbench_metrics.clip` / `.clip_iqa` / `.imagereward`。

### 3.2 GenEval（独立进程）

```bash
export CUDA_VISIBLE_DEVICES=1
/home/kaokaolv/.venv-omini-latest/bin/python -m aqa.cli run \
  examples/repro_zimage_mxfp8_geneval.yaml
```

必须单独进程：同进程第二次 `Omni` 加载在本栈上曾生成全黑图（仓内先例见
`examples/run_zimage_turbo_mxfp8_geneval_only.yaml`）。2212 张图约 75 min（约 2 s/图 + 判分），结果键 `geneval`。

### 3.3 查看结果

```bash
python3 - <<'PY'
import json
for p in [".../Z-Image-Turbo-MXFP8-RTN-20261024-drawbench/eval/accuracy.json",
          ".../Z-Image-Turbo-MXFP8-RTN-20261024-geneval/eval/accuracy.json"]:
    d=json.load(open(p)); print(p)
    for k,v in d["results"].items(): print(" ", k, round(v["score"],4), v["metric"], v["num_samples"])
PY
```

耗时参考：模型加载 ~2 min/次；DrawBench 生成 200 图 ~10 min，重打分（复用图片）~2 min；GenEval 74.5 min。

## 4. 已知坑

1. **ImageReward 需要 `bert-base-uncased` tokenizer**。离线模式且未缓存时会报
   `OSError: Can't load tokenizer for 'bert-base-uncased'`，整个打分阶段失败（图片已生成，重跑会按 `metadata.csv` 跳过生成）。按 §2 准备缓存 + 别名即可。
2. **`hf download bert-base-uncased` 会 404**：直接下 `google-bert/bert-base-uncased`，再建 `models--bert-base-uncased` 符号链接（离线解析用别名）。
3. **GenEval 单独进程**（见 3.2），否则有全黑图风险。
4. **日志尾部 `TypeError: argument should be integer or bytes-like object, not 'str'`**：出现在结果写出之后，来自 vllm 的 stdout 前缀补丁与 click 的交互；`aqa_run.json` 仍为 `done`、`accuracy.json` 完整，属噪声。
5. 评估结束后的 `model_card.md` 只在量化 run 写；eval-only run 不生成模型卡。

## 5. 产物位置（本机）

| 路径 | 内容 |
|---|---|
| `/home/kaokaolv/models/Z-Image-Turbo-MXFP8-RTN-AutoRound` | 已发布 checkpoint（14.7 GB） |
| `/home/kaokaolv/my_commit/aqa-runs/Z-Image-Turbo-MXFP8-RTN-20261024-drawbench/` | CLIP/CLIP-IQA/ImageReward + 200 张图 |
| `/home/kaokaolv/my_commit/aqa-runs/Z-Image-Turbo-MXFP8-RTN-20261024-geneval/` | GenEval + 2212 张图 |
| `/home/kaokaolv/my_commit/aqa-runs/Z-Image-Turbo-MXFP8-RTN-20261024-repro/report.md` | 复现报告 |

## 6. 如果要连量化一起复现（本次未做）

需要 bf16 底模 `Tongyi-MAI/Z-Image-Turbo`（约 31 GB）与 auto-round diffusion 环境，配方：

```yaml
quant:
  scheme: MXFP8
  method: autoround
  export_format: auto_round
  iters: 0
  extra_args:
    - --disable_opt_rtn
    - --disable_model_free
    - --ignore_layers
    - adaLN_modulation
    - --dataset
    - coco2014
    - --num_inference_steps
    - "8"
    - --guidance_scale
    - "0.0"
```

完整示例见 `examples/run_zimage_turbo_mxfp8.yaml`（把 `model` 改成本机底模路径），量化约 20 min，然后再按 §3 评估。注意：重新量化得到的权重与已发布 checkpoint 不完全相同，分数按容差判断（建议 CLIP ±0.3、GenEval ±0.01）。

## 7. 另一条打分路径：neural-compressor 的 `flux/main.py`

除 aqa 自带 diffusion harness 外，也可以用 neural-compressor 仓库的评估脚本对**已生成好的**图片打分：

```
/home/kaokaolv/aqa/neural-compressor/examples/pytorch/diffusion_model/flux/main.py
```

脚本的 `--accuracy` 模式约定：

- `--eval_dataset`：TSV，必须含 `id` 与 `caption` 两列；
- `--output_image_path`：图片目录，文件名必须是 `<id>.png`（与 TSV 的 `id` 一一对应）；
- 指标：`clip` / `clip-iqa` / `imagereward`（`auto_round.compressors.diffusion.eval.metric_map`）。

本机 `.venv-omini-latest` 没有安装 `neural_compressor`，且其自带的 auto_round 0.17 不含
`compressors/diffusion`，因此运行时要显式加 `PYTHONPATH` 指向两处源码：

```bash
# 0) 变量
IMG=/home/kaokaolv/my_commit/aqa-runs/Z-Image-Turbo-MXFP8-RTN-20261024-drawbench/eval/drawbench_metrics

# 1) 从已生成的 metadata.csv 造 eval TSV（id 0..199, caption=prompt）
/home/kaokaolv/.venv-omini-latest/bin/python - <<'PY'
import pandas as pd
img = "/home/kaokaolv/my_commit/aqa-runs/Z-Image-Turbo-MXFP8-RTN-20261024-drawbench/eval/drawbench_metrics"
df = pd.read_csv(f"{img}/images/metadata.csv")
df.rename(columns={"idx": "id", "prompt": "caption"})[["id", "caption"]].to_csv(
    f"{img}/drawbench_eval.tsv", sep="\t", index=False)
print("rows:", len(df))
PY

# 2) 建 <id>.png 视图（0.png -> 0000.png）
mkdir -p $IMG/images_by_id
for i in $(seq 0 199); do
  ln -sf $IMG/images/$(printf '%04d' $i).png $IMG/images_by_id/$i.png
done

# 3) 跑 neural-compressor 的 accuracy 打分
cd /home/kaokaolv/aqa/neural-compressor/examples/pytorch/diffusion_model/flux
PYTHONPATH=/home/kaokaolv/my_commit/auto-round:/home/kaokaolv/aqa/neural-compressor \
CUDA_VISIBLE_DEVICES=0 \
/home/kaokaolv/.venv-omini-latest/bin/python main.py \
  --accuracy \
  --eval_dataset $IMG/drawbench_eval.tsv \
  --output_image_path $IMG/images_by_id
```

注意：

- `PYTHONPATH` 顺序要让 `my_commit/auto-round` 先于 site-packages，才能用到带
  `compressors/diffusion` 的 auto_round；`aqa/neural-compressor` 提供 `neural_compressor` 包。
- `main.py` 顶部会 import `neural_compressor.torch.quantization`、diffusers、Flux 等，
  即使只跑 `--accuracy` 也会走这些 import；缺依赖时会直接 ImportError。
- 这条路径与 aqa 的 `drawbench_metrics` 实现不同（各自的 CLIP/ImageReward 封装），
  分数可作交叉验证，但不要混用两套数字做严格 A/B。

### 7.1 本栈实际可运行的版本（2026-10-26 实测）

裸跑 `main.py` 在 `.venv-omini-latest` 上会依次撞三个问题：

1. 缺 `torchmetrics`（CLIP / CLIP-IQA）：`ModuleNotFoundError: No module named 'torchmetrics'`；
2. transformers 5.14 的 `CLIPModel.get_{image,text}_features` 返回
   `BaseModelOutputWithPooling`，torchmetrics 1.9 期望投影后的 tensor：
   `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`；
3. `image-reward` 1.5 与 transformers 5.x 不兼容（v4 符号被删）。

处理方式：

- 已向 `.venv-omini-latest` 安装缺失包（只新增、不升级）：
  `torchmetrics==1.9.0`、`open-clip-torch==3.3.0`、`timm==1.0.30`、
  `lightning-utilities==0.15.3`；
- 用兼容 wrapper `/home/kaokaolv/my_commit/zimage_nc_shim/run_accuracy_compat.py`：
  运行前把 `CLIPModel.get_{image,text}_features` 的返回值解包为 `.pooler_output`，
  并把 `metric_map["imagereward"]` 换成 aqa 的隔离 transformers-4.44 子进程
  （`aqa.eval.image_metrics.imagereward_score`，模型同为 ImageReward-v1.0）。

实际执行的命令（在 §7 的前两步 TSV / 软链之后）：

```bash
cd /home/kaokaolv/aqa/neural-compressor/examples/pytorch/diffusion_model/flux
PYTHONPATH=/home/kaokaolv/my_commit/auto-round:/home/kaokaolv/aqa/neural-compressor \
HF_HOME=/home/kaokaolv/hf_home_aqa \
CUDA_VISIBLE_DEVICES=0 \
/home/kaokaolv/.venv-omini-latest/bin/python \
  /home/kaokaolv/my_commit/zimage_nc_shim/run_accuracy_compat.py \
  --accuracy \
  --eval_dataset $IMG/drawbench_eval.tsv \
  --output_image_path $IMG/images_by_id
```

实测输出（200 张 DrawBench 图，耗时约 2 分 14 秒）：

```
+-------------+-----------+
| CLIP        | 27.7619   |
+-------------+-----------+
| CLIP-IQA    |  0.918089 |
+-------------+-----------+
| ImageReward |  0.982469 |
+-------------+-----------+
```

口径对比（**不要混用绝对值**）：

| 指标 | NC 脚本（本次） | aqa `drawbench_metrics` | Card |
|---|---:|---:|---:|
| CLIP | 27.762（`openai/clip-vit-large-patch14`） | 31.774（`clip-vit-base-patch32`） | 31.79 |
| CLIP-IQA | 0.918（torchmetrics CLIP-IQA，0..1） | 70.577（反义 prompt 对，0..100） | 70.41 |
| ImageReward | 0.9825（隔离子进程） | 0.9825 | 0.97 |

- CLIP 差异来自模型不同（ViT-L/14 vs ViT-B/32），不是实现错误；
- CLIP-IQA 的标尺不同（0..1 vs ×100），且 prompt 对/softmax 实现不同；
- ImageReward 两套完全一致，因为 wrapper 直接复用 aqa 的隔离 ImageReward 实现；
- 因此 NC 脚本适合做**同一实现内部的交叉验证**，与 card 的对应关系应看 aqa 那列
  （card 的数字与 aqa 的 ViT-B/32 实现一致）。

### 7.2 为什么 CLIP-IQA 与 aqa/card 对不齐（隔离实验）

在**同一批 200 张图**上分别换 backbone 与 prompt 集合，结果（统一 ×100 显示）：

| 实现 | Backbone | Prompt 集合 | CLIP-IQA |
|---|---|---|---:|
| aqa | ViT-B/32 | 3 对（quality / good quality / aesthetic） | **70.577** |
| torchmetrics | ViT-B/32 | 1 对（`quality`，NC 脚本默认） | 83.180 |
| torchmetrics | ViT-B/32 | 3 对 | 70.188（83.18 / 92.99 / 34.40） |
| aqa | ViT-L/14 | 3 对 | **76.826** |
| torchmetrics | ViT-L/14 | 1 对（`quality`） | 91.809 |
| torchmetrics | ViT-L/14 | 3 对 | 76.422（91.81 / 96.42 / 41.04） |

结论（按影响大小）：

1. **Prompt 集合是主因**。torchmetrics 默认 `prompts=("quality",)`，只算
   "Good photo." vs "Bad photo."；aqa 用 3 对，其中
   "Aesthetically pleasing." vs "Aesthetically displeasing." 得分很低
   （ViT-B/32 下 34.4、ViT-L/14 下 41.0），把 3 对均值从 ~83 拉到 ~70。
2. **Backbone 次因**。同样 3 对时，ViT-L/14 比 ViT-B/32 高约 6.2 分
   （76.83 vs 70.58）。
3. **标尺**。torchmetrics 返回 0..1，aqa/card 是 ×100；乘 100 后可直接比。
4. **实现本身一致**：同 backbone + 同 3 对时，aqa 70.577 vs torchmetrics 70.188，
   差 0.39（残差来自 aqa 用 CLIP 的 `logits_per_image`（学习到的 logit_scale）
   vs torchmetrics 固定 ×100 cosine）。

因此：

- NC 脚本输出的 `0.918`（=91.8）是 **ViT-L/14 + 单 quality prompt**；
- card/aqa 的 `70.41/70.58` 是 **ViT-B/32 + 3 对 prompt 均值**；
- 两条数字都对，只是不是同一个 CLIP-IQA 变体。要严格对齐 card，需让
  torchmetrics 用 ViT-B/32、传入同样的 3 对 prompt 并 ×100；但注意
  `compute().mean()` 只支持单 prompt（多 prompt 返回 dict，NC 脚本会报错），
  所以脚本原样跑时只能出单 prompt 版本。
- 同一隔离实验也解释了 CLIP 的差异：NC 用 `clip-vit-large-patch14`（27.76），
  aqa/card 用 `clip-vit-base-patch32`（31.77）；换同一 backbone 后二者一致。

### 7.3 为什么 aqa 的 CLIP-IQA 用 3 对 prompt 均值

定义位置：`aqa/eval/image_metrics.py:73-77`（`CLIP_IQA_PROMPT_PAIRS`），代码注释写明
"CLIP-IQA paper style"：

| 维度 | Positive | Negative |
|---|---|---|
| 整体观感 | Good photo. | Bad photo. |
| 技术质量 | Good quality. | Bad quality. |
| 美学 | Aesthetically pleasing. | Aesthetically displeasing. |

设计原因：

1. **降低 prompt 敏感性**。CLIP-IQA 原论文指出 prompt 模板/措辞对分数影响显著
   （[Wang et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25353/25125)），
   单对 prompt 方差大；对多对取均值是这类工作的常见做法（多 antonym-prompt 变体，
   如 [arXiv 2308.13094](https://arxiv.org/pdf/2308.13094)）。
2. **覆盖不同质量维度**。只测 "Good photo." / "Bad photo." 更偏“照片质量”，对
   T2I 更关心的美学不敏感。§7.2 的隔离实验给出证据：单 quality 对在 ViT-B/32 下
   83.18、ViT-L/14 下 91.81，而 aesthetic 对只有 34.40 / 41.04；只取 quality 会
   系统性偏高，3 对均值把美学拉进来。
3. **不用 torchmetrics 的 16 对**。其内置维度里很多（scary / lonely / new / happy…）
   不是 T2I 质量维度，全平均会引入无关噪声；默认又只取 1 对，太窄。3 对是
   “覆盖够用、噪声可控”的折中。

边界：这是 aqa 的 benchmark 固定定义（单一真源
`recipes/benchmarks/drawbench_metrics.yaml`），card 的 70.41 与项目内所有历史
数字都按这套口径，因此内部可比；它不是“标准 CLIP-IQA”的唯一实现，跨工具比较
必须同时对齐 **backbone + prompt 集合 + 标尺**（见 §7.2 矩阵）。要与 NC 脚本对齐，
需用 ViT-B/32 + 同样 3 对 + ×100；脚本原样跑只能输出单 quality prompt 版本
（`compute().mean()` 不支持多 prompt 的 dict 返回）。

## 附：复现用 yaml

`examples/repro_zimage_mxfp8_drawbench.yaml`：

```yaml
model: /home/kaokaolv/models/Z-Image-Turbo-MXFP8-RTN-AutoRound
eval:
  benchmarks: [drawbench_metrics]
  backend: vllm_omni
  harness: diffusion
  baseline: false
  diffusion:
    num_inference_steps: 8
    guidance_scale: 0.0
    height: 1024
    width: 1024
    seed: 42
    scorer_model: openai/clip-vit-base-patch32
agent: {fix_loop: false, optimize: false}
executor: {type: local, num_gpus: 1}
output:
  dir: /home/kaokaolv/my_commit/aqa-runs/Z-Image-Turbo-MXFP8-RTN-20261024-drawbench
  unique_dir: false
  report_model: Tongyi-MAI/Z-Image-Turbo
  artifact_name: Z-Image-Turbo-MXFP8-RTN-AutoRound
  sinks: [{type: local_fs}]
```

`examples/repro_zimage_mxfp8_geneval.yaml` 仅把 `benchmarks` 换成 `[geneval]`、去掉 `scorer_model`、`output.dir` 换成 geneval 目录。
