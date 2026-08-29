<!-- Distilled auto_round diffusion/VLM recipe knowledge. Embedded into the
agent-optimize prompt ONLY when the model is detected as a diffusion pipeline
(model_index.json present). Full source-verified version (all file:line
citations): docs/autoround_diffusion_reference.md in this repo. -->

# AutoRound Diffusion / 多模态量化 Recipe 要点

> 来源：`auto-round` @ `bd9a13ce`（2026-08-25 重构版）。完整版含全部 `文件:行号` 出处与外部链接，见仓内 `docs/autoround_diffusion_reference.md`。

## 1. 架构约束（最重要）

- **只量化 `pipe.transformer`（DiT 系：FLUX、SD3、Z-Image）或 `pipe.unet`（SD/SDXL）**；text encoder（T5/CLIP/Mistral-VLM）、VAE 一律不量化，保存时原样 `save_pretrained`。
- 多 transformer 模型（Wan2.2 的 `transformer`+`transformer_2`）逐一量化。
- 模型经根目录 `model_index.json` 自动识别为 diffusion，走统一 `AutoRound` 入口（`AutoRoundDiffusion` 是 deprecated 别名）。
- 支持矩阵官方列出：FLUX.1-dev、**Z-Image / Z-Image-Turbo**、NextStep-1.1、Ovis-Image-7B、stable-audio-open-1.0、Wan2.2 系列、Cosmos3。其余 diffusers pipeline 走通用路径（能否跑通取决于是否符合标准结构）。

## 2. 校准

- 默认数据集 **`coco2014`**（captions tsv；LLM 默认的 pile-10k 会被强制换掉）；本地 `.tsv` 需 `id`+`caption` 两列。
- **`batch_size` 强制为 1**，不要尝试调大。
- 校准代价 = `nsamples × num_inference_steps` 次完整去噪前向（整条 pipeline 真跑）。大 DiT（FLUX.1/FLUX.2-dev）**先 `iters 0`（RTN）起步**，冒烟用 `iters 1 / nsamples 1 / steps 2`。
- `--num_inference_steps` / `--guidance_scale` / `--generator_seed` 按模型推理惯例设置（例：Z-Image-Turbo 是 8 步、guidance 0.0；FLUX.1-dev 是 28 步、guidance 3.5）。

## 3. 导出与 vllm-omni 部署

- **`fake` 格式不能重新加载**，仅供进程内精度对比；要部署必须落盘为真实格式。
- **⚠️ RTN 的 model_free 自动路由（实跑踩坑确认）**：`--iters 0 --disable_opt_rtn` 会被自动路由到 model-free 路径（日志 "Auto-routing to model-free quantization..."）。该路径对 diffusion 模型会**静默空量化**（0 层：transformer config 无 `model_type` 导致 layer 检测失败 + shard 预取错找仓库根目录 404），却仍写入 quantization_config——导出的是「假量化」checkpoint。diffusion 跑 RTN **必须加 `--disable_model_free`**，并检查日志 `Quantized layers (N)` 的 N>0。
- **⚠️ 导出可能损坏非量化组件（实跑踩坑确认）**：auto-round 会自行重存 text encoder 等组件，可能破坏权重名——Z-Image 的 Qwen3 text encoder 导出后所有权重丢了内层 `model.` 前缀（`model.layers.N.*` → `layers.N.*`），vllm-omni 加载报 `no module or parameter named 'text_encoder.embed_tokens'`。aqa 量化后自动做 post-export repair（比对源模型键名、不一致就从源目录复制还原，产 `aqa_export_repair.json`）。
- **⚠️ 引擎侧全精度保留层不能被量化（实跑踩坑确认）**：vllm-omni 的 DiT 实现把精度敏感小层写死 `quant_config=None`（如 Z-Image 的 `adaLN_modulation`、`t_embedder.mlp`）。auto-round 会按 block 把它们一起量化，导致加载时键型不匹配（`KeyError: ...adaLN_modulation.0.qweight`）。量化时用 `--ignore_layers adaLN_modulation` 排除；通用做法：查 vllm-omni 对应模型文件中 `quant_config=None` 的模块并对齐忽略清单。
- vllm-omni 部署**必须 `--format auto_round`**。支持矩阵：

| scheme | vllm-omni 状态 |
|---|---|
| W4A16（INT4） | ✅ 已合并（PR #1777），FLUX.1-dev/Wan2.2/GLM-Image/Cosmos3 已验证 |
| MXFP8 | ✅ 代码支持（PR #3782，`IncMxfp8OfflineLinearMethod`） |
| MXFP4 | ✅ 上游 main 已合入（PR #5544，2026-08-25）；早于合并点的本地快照会拒载 |

- **checkpoint 前提**：transformer 子目录 `config.json` 的 `quantization_config` 必须含 `quant_method="auto-round"` + `block_name_to_quantize`（如 `"transformer_blocks,single_transformer_blocks"`）。auto-round 导出是否自动写入需实跑确认，缺失则后处理补写该 JSON 字段。
- `svdquant_nunchaku` 仅 FLUX.1 + MXFP4 端到端验证过；其 adapter `auto` 检测类名子串 `"fluxtransformer"`，**FLUX.2（`Flux2Transformer2DModel`）不命中** → nunchaku 导出被拒，FLUX.2 当前不能走 SVDQuant。

## 4. 逐模型要点

- **Z-Image-Turbo ✅**（支持度最高）：官方支持矩阵 + 有单测，纯通用路径。推理惯例 8 步、guidance 0.0。
- **FLUX.2-dev / klein-4B / klein-9B ⚠️**（通用路径，未验证）：
  1. 先确认 block 类名——auto-round 只对 FLUX.1 的 `FluxTransformerBlock`/`FluxSingleTransformerBlock` 注册了双输出 block 处理，Flux2 若类名不同且返回双输出，需在 `block_runner.py` 补注册；
  2. text encoder 是 24B Mistral-VLM（不量化但参与校准前向），dev 版整套 bf16 权重 110GB+，校准显存/耗时巨大——**冒烟顺序 klein-4B → klein-9B → dev**；
  3. 不能走 SVDQuant（见上）。
- **DeepSeek-OCR-2**：不是 diffusion，走通用 MLLM 文本量化路径（只量化文本塔），GGUF 导出明确支持。
- **FLUX.1-dev ✅**：唯一有端到端部署导出的基准模型；aqa 全链路示例见 `examples/run_diffusion_flux1.yaml`。

## 5. 命令模板（DiT → vllm-omni）

```bash
auto-round --model_name <org/model> \
  --scheme W4A16 --format auto_round \
  --iters 0 --disable_opt_rtn --disable_model_free \
  --dataset coco2014 --nsamples 128 \
  --num_inference_steps 28 --guidance_scale 3.5 \
  --output_dir <out>
```

- **`--model_name` 必须是本地目录**（diffusion 常规加载直接 `open(<path>/model_index.json)`，无 hub 解析）——先 `hf download <id> --local-dir <path>`。
- **`--disable_model_free` 必加**：`iters 0 + disable_opt_rtn` 会被自动路由到 model_free，该路径对 diffusion 会静默空量化（0 层）。

- scheme 三选一：W4A16 / MXFP8 / MXFP4（PR #5544 已合入上游 main，均可部署；早于合并点的本地快照 MXFP4 会拒载）。
- 追求精度再开 `--iters 200`（校准代价随之 ×1 完整去噪/样本）。
