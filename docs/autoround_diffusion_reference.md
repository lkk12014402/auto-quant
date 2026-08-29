# AutoRound Diffusion / 多模态模型量化参考（源码核实版）

> 来源：`/home/kaokaolv/auto_agent/auto-round` @ `bd9a13ce`（2026-08-25）。
> 与《AutoRound_算法与配置全参考.md》配套，本文只覆盖 **diffusion（文生图/视频/音频）与 VLM**。
> 结论全部带 `文件:行号`；模型架构事实附外部来源链接。

## 0. 五个目标模型速览

| 模型 | 类型 | auto-round 能否量化 | 路径 | 主要风险 |
|---|---|---|---|---|
| Z-Image-Turbo（Tongyi-MAI，6B S3-DiT） | 文生图 DiT | ✅ 官方支持矩阵 + 有单测 | 通用 DiffusionPipeline → 量化 `pipe.transformer` | fake 仅供研究；W4A16 + auto_round 格式可给 vLLM-Omni（Z-Image 未官方验证） |
| FLUX.2-dev（BFL，32B DiT） | 文生图 DiT | ⚠️ 通用路径理论可行，未验证 | 同上 | Flux2 block 类名 ≠ FLUX.1，双输出 block 注册可能不命中；校准显存需求大 |
| FLUX.2-klein-4B / 9B（BFL 蒸馏版） | 文生图 DiT | ⚠️ 同上，规模更小更适合先试 | 同上 | 同上 |
| DeepSeek-OCR-2（DeepEncoder V2 + 3B LM） | VLM（非 diffusion） | ⚠️ 走通用 MLLM 文本量化路径（无专属适配）；GGUF 导出明确支持 | MLLM 路径只量化文本塔 | template fallback default，processor 兼容性需试 |
| （参考）FLUX.1-dev | 文生图 DiT | ✅ 唯一有端到端部署导出（SVDQuant→Nunchaku） | 通用路径 + SVDQuant | — |

## 1. Diffusion 量化架构（代码事实）

**入口与路由**：
- 没有独立的 diffusion 类：`AutoRoundDiffusion` 是 deprecated 别名（`autoround.py:869-872`），统一走 `AutoRound`。
- 模型类型检测：`is_diffusion_model`（`utils/model.py:1290-1334`）——依据 `model_index.json` 存在 / `DiffusionPipeline` 实例 / `model_type=="nextstep"` 特例。路由由 `DiffusionMixin`（`compressors/diffusion_mixin.py:33`）+ `DiffusionCalibrator`（`calibration/diffusion.py:51`）注入。

**量化目标（最重要的架构约束）**：
- **只量化 `pipe.unet`（SD/SDXL）或 `pipe.transformer`（DiT 系：Flux、SD3、Z-Image）**（`utils/model.py:959-964`）。
- **text encoder（T5/CLIP/Mistral-VLM）、VAE 一律不量化**，只做 dtype 对齐（`diffusion_mixin.py:126-138`），保存时原样 `save_pretrained`（`diffusion_mixin.py:592-594`）。
- 多 transformer（如 Wan2.2 的 `transformer`+`transformer_2`）会逐一量化（`diffusion_mixin.py:150-165, 397-501`）。

**校准**：
- 真的逐条 prompt 跑 pipeline 去噪（hooks 捕获 block 输入；代价 = `nsamples × num_inference_steps` 次前向，`calibration/diffusion.py:98-213`）。`batch_size` 强制为 1（`diffusion_mixin.py:86-113`）。
- 校准数据集：默认 `coco2014`（captions tsv，`compressors/diffusion/dataset.py:29-34`；LLM 的 pile-10k 会被强制换掉，`diffusion_mixin.py:71-73`）；`audiocaps`（音频模型）；本地 `.tsv` 需 `id`+`caption` 两列（`dataset.py:76-81`）。
- CLI 参数：`--guidance_scale`(7.5)、`--num_inference_steps`(50)、`--generator_seed` 进入量化；`--prompt/--prompt_file/--metrics(clip/clip-iqa/imagereward)/--image_save_dir` 只用于量化后评估（`cli/parser.py:278-290`、`eval/evaluation.py:130-184`）。

**导出与部署**：
- `fake` 格式：量化后**不支持重新加载**（`compressors/diffusion/README.md:12`；评估是把内存里的量化 transformer 塞回 pipe，`eval/evaluation.py:139-147`）。仅用于进程内精度对比。
- **`auto_round` 格式（W4A16/MXFP8）→ vLLM-Omni 可部署**（详见第 5 节）：PR [#1777](https://github.com/vllm-project/vllm-omni/pull/1777) 已合并，FLUX.1-dev、Wan2.2、GLM-Image、Cosmos3 等已验证。
- `svdquant_nunchaku`：仅 FLUX.1 + MXFP4 端到端验证过（`docs/svdquant_details.md:16-18, 238`），加载方 Nunchaku + Diffusers。

## 2. 官方支持矩阵 vs 通用路径

代码里有专门处理的：FLUX.1（双输出 block 注册 `algorithms/block_runner.py:73-74`）、SD/SDXL（unet）、SD3 类 DiT、Wan2.x（双 transformer）、Stable Audio、NextStep-1.1（专用 pipeline）、Cosmos3（专用加载，导出对齐 vllm-omni）。

文档支持矩阵（`compressors/diffusion/README.md:59-76`）：FLUX.1-dev、**Tongyi-MAI/Z-Image 与 Z-Image-Turb**、NextStep-1.1、Ovis-Image-7B、stable-audio-open-1.0、Wan2.2 系列、Cosmos3-Nano/Super。

其余任何 diffusers pipeline 都走**通用路径**（`DiffusionPipeline.from_pretrained` + unet/transformer 启发式），能不能跑通取决于 pipeline 是否符合 diffusers 标准结构。

## 3. SVDQuant 专项（FLUX.1 唯一部署路径）

- 定位：结构预处理器，`W ≈ Q(R) + U@V`（低秩分支吸收 outlier，残差量化到 MXFP4 W4A4）。CLI：`--algorithm svdquant`（自动强制 `format=svdquant_nunchaku`，`autoround.py:676-678`），要求 nblocks=1。
- `--svdquant-model-adapter auto/identity/flux`：`auto` 检查类名是否含 **"fluxtransformer"**，是→flux 否则→identity（`export/svdquant_adapters/__init__.py:58-59`）；`svdquant_nunchaku` 导出**拒绝 identity**（`export/formats/backends/svdquant_nunchaku.py:167-171`）。
- ⚠️ **FLUX.2 的坑**：`Flux2Transformer2DModel` 小写后是 `flux2transformer...`，**不含子串 `fluxtransformer`** → `auto` 会落到 `identity` → nunchaku 导出被拒。FLUX.2 要走 SVDQuant 需新增 adapter（映射 FLUX.2 的模块名到 Nunchaku schema），当前代码没有。
- 产物：`transformer/diffusion_pytorch_model.safetensors` + pipeline 目录；加载方为 **Nunchaku**（`NunchakuFluxTransformer2dModel`）+ Diffusers `FluxPipeline`（`docs/svdquant_details.md:202-234`）。推理才需要装 nunchaku。

## 4. 逐模型评估

### 4.1 Z-Image-Turbo ✅（支持度最高）

- 事实：Tongyi-MAI 6B 单流 DiT（S3-DiT），8-NFE 蒸馏（[GitHub README](https://github.com/Tongyi-MAI/Z-Image)）。
- auto-round 官方支持矩阵明确列出；单测 `test_z_image_tune`（`test/unit/test_cuda/models/test_diffusion.py:52-61`）+ tiny fixture（`test/fixtures.py:153-179`）。无任何专用代码，走通用路径 → 说明通用路径对它已验证。
- 试跑命令（冒烟用 iters=1/nsamples=1，正式再加）：
```bash
auto-round --model_name Tongyi-MAI/Z-Image-Turbo \
  --scheme W4A16 --iters 200 --nsamples 128 \
  --num_inference_steps 8 --guidance_scale 0.0 \
  --format fake --output_dir ./Z-Image-Turbo-W4A16
```
  注意：num_inference_steps/guidance 按该模型推理惯例设（Turbo 是 8 步、guidance 0）；校准时会真跑 8 步 × nsamples 次去噪。
- scheme 选择：测试用过 MXFP4；W4A16/MXFP8 亦可（通用 scheme 校验，无 diffusion 专属限制）。导出：`fake` 仅供进程内精度对比；**`--format auto_round` + W4A16 可给 vLLM-Omni 部署**（见第 5 节，Z-Image 需 vllm-omni 侧架构支持，Int8 路径已验证过 Z-Image 说明架构已支持）。

### 4.2 FLUX.2-dev / FLUX.2-klein-4B / 9B ⚠️（通用路径，未验证）

- 事实：FLUX.2 是 BFL 2025-11 发布的 32B rectified-flow DiT（[MarkTechPost](https://www.marktechpost.com/2025/11/25/black-forest-labs-releases-flux-2-a-32b-flow-matching-transformer-for-production-image-pipelines/)），diffusers `Flux2Pipeline` + `Flux2Transformer2DModel`，text encoder 换成 **Mistral-Small-3.2-24B VLM**（不再是 T5+CLIP，[prompt guide](https://github.com/bako02/flux2-prompt-guide)）。klein-4B/9B 是 2026-01 的蒸馏紧凑版（[NVIDIA NIM](https://build.nvidia.com/black-forest-labs/flux_2-klein-4b/modelcard)）。
- 通用路径匹配性：`model_index.json` 存在 → 判为 diffusion；`pipe.transformer` 存在 → 量化目标正确。text encoder（24B Mistral）按架构不量化。
- **三个必须先核实的点**：
  1. **block 类名**：auto-round 只对 `FluxTransformerBlock`/`FluxSingleTransformerBlock`（FLUX.1）注册了"双输出 block"处理（`block_runner.py:73-74`）。Flux2 的 transformer_blocks/single_transformer_blocks 若类名不同且 block 返回 `(hidden, encoder_hidden)` 双输出，校准 hook 可能处理不了 → 需要在 `block_runner.py` 补注册。先打印 `type(pipe.transformer.transformer_blocks[0])` 确认。
  2. **SVDQuant 不可自动用**：adapter `auto` 检测子串 `fluxtransformer`，Flux2 不命中（第 3 节）→ 导出 nunchaku 会被拒。当前只能用 fake。
  3. **校准资源**：校准要完整跑 pipeline（32B transformer + 24B text encoder + VAE，bf16 约 110GB+ 权重），`nsamples × num_inference_steps` 次前向。dev 版 50 步校准代价巨大——先试 klein-4B（约 8GB）把流程跑通。
- 冒烟顺序建议：klein-4B（iters=1, nsamples=1, steps=2）→ klein-9B → dev。

### 4.3 DeepSeek-OCR-2 ⚠️（不是 diffusion，走 MLLM 文本路径）

- 事实：DeepEncoder V2 视觉编码器 + 3B 语言解码器的 VLM（[arXiv 2601.20552](https://arxiv.org/abs/2601.20552)）。
- auto-round 中**无任何 deepseek_ocr 专属适配**（`SUPPORT_ONLY_TEXT_MODELS`/`SPECIAL_MULTIMODAL_BLOCK`/templates 均无，`special_model_handler.py:36-56, 591-599`）。
- 可行路径：通用 MLLM 文本量化——`is_mllm_model` 识别（`utils/model.py:1222-1274`）→ 只量化文本塔（默认 `quant_nontext_module=False`），template 未注册时 fallback `TEMPLATES["default"]`（`template.py:185-188`）。风险在 processor/chat-template 兼容性，需试跑。
- **GGUF 导出明确支持**：`DeepseekOCR2ForCausalLM` 注册于 `export/export_to_gguf/conversion/deepseek.py:109-116`（vision tower 作 mmproj 导出）。若目标是 llama.cpp 部署，这是官方验证过的路。
- 试跑：`auto-round --model_name deepseek-ai/DeepSeek-OCR-2 --scheme W4A16 --iters 200 --format auto_round`（视觉塔默认自动保护）。

## 5. vLLM-Omni 推理（auto_round 格式部署路径）

> 来源：本地 `/home/kaokaolv/auto_agent/vllm-omni` main @ `51b7565f`（2026-08-25）+ PR [#1777](https://github.com/vllm-project/vllm-omni/pull/1777)（已合并，commit `c1d2dcc3`）、PR [#5544](https://github.com/vllm-project/vllm-omni/pull/5544)（**已于 2026-08-25 合并进上游 main**；本地 main 快照早于此，用上游 main 或后续 release 即可）。

**diffusion 模型用 auto-round 导出的 `auto_round` 格式可以直接给 vLLM-Omni 推理**：vLLM-Omni 读 checkpoint 的 `config.json`，自动检测 `quantization_config.quant_method="auto-round"`，无需 `--quantization` 参数（`docs/user_guide/quantization/autoround.md`）。

### 5.1 AutoRound scheme 支持现状（vllm-omni 侧）

| scheme | 状态 | 代码路径 | 已验证模型 |
|---|---|---|---|
| **W4A16**（INT4, g128, sym, `auto_round:auto_gptq`） | ✅ 已合并已验证 | `OmniINCConfig` → 标准 INC 路径（GPTQ-Marlin / XPU kernel） | FLUX.1-dev、Wan2.2-I2V/T2V/TI2V、GLM-Image、Cosmos3-Nano/Super、Qwen-Image（e2e 测试在 `tests/e2e/offline_inference/test_*_autoround_w4a16_expansion.py`） |
| **MXFP8**（`data_type="mx_fp"`, bits=8） | ✅ 代码已支持（验证面窄） | `OmniINCConfig.get_quant_method` → `IncMxfp8OfflineLinearMethod`（`vllm_omni/quantization/inc_config.py:66-84,180-277`）；支持 packing `auto_round:auto_gptq/auto_awq/llm_compressor` | 未见公开验证模型 |
| **MXFP4**（`data_type="mx_fp"`, bits=4） | ✅ **上游 main 已支持**（PR #5544 于 2026-08-25 合并） | PR 扩展了 `OmniINCConfig` 对 `weight_bits==4` 的 mx_fp 识别（Wan2.1-T2V + FLUX.1-dev，XPU 路径）；本地快照 `51b7565f` 之前的 `inc_config.py:76` 只认 bits==8 | PR 里验证了 Wan2.1-T2V-1.3B 和 FLUX.1-dev |
| NVFP4 / 其他 | ❌ 无 auto-round 路径 | — | — |

注意区分：vllm-omni 树里的 `mxfp4_config.py`（`quant_method="mxfp4"/"mxfp4_dualscale"`）是**另一套非 AutoRound 的原生 MXFP4 路径，仅 NPU/ROCm**（`mxfp4_config.py:94-170`），与 auto-round 导出的 MXFP4 checkpoint 无关。

### 5.2 checkpoint 要求

量化产物（transformer 子目录的 `config.json`）须含：

```json
{"quantization_config": {"quant_method": "auto-round", "bits": 4, "group_size": 128,
  "sym": true, "packing_format": "auto_round:auto_gptq",
  "block_name_to_quantize": "transformer_blocks,single_transformer_blocks"}}
```

`block_name_to_quantize` 用于把 checkpoint 的 block 名映射到 vLLM-Omni 运行时模块名（`inc_config.py:86-105` 的 remap 逻辑）。

### 5.3 官方量化命令（autoround.md 示例）

```bash
auto-round \
  --model black-forest-labs/FLUX.1-dev \
  --scheme W4A16 \
  --batch_size 1 \
  --disable_opt_rtn \
  --dataset coco2014 \
  --iters 0
# 输出目录直接作为 vLLM-Omni 的 model 参数
```

推理：`Omni(model="<输出目录>")` 或 `examples/offline_inference/text_to_image/text_to_image.py --model <输出目录>`。

### 5.4 硬件支持（autoround.md 表格）

NVIDIA SM 80+（Ampere 起）✅、Intel XPU ✅、AMD ROCm 未验证、Ascend NPU ❌。

### 5.5 对第 4 节各模型的影响

- **FLUX.1-dev**：W4A16 全链路已通（量化→导出→vllm-omni 推理），公开 checkpoint `vllm-project-org/FLUX.1-dev-AutoRound-w4a16`。
- **Z-Image / FLUX.2 系 / Qwen-Image 等**：W4A16 是 checkpoint-driven 的通用机制（"A model is supported when its checkpoint uses a compatible INC/AutoRound config and the target stage maps to vLLM-Omni's runtime module names"），理论上任何 DiT 都能走通，但**需要 vllm-omni 已支持该模型架构**（FLUX.2 需要 vllm-omni 侧的 Flux2 模型实现；Qwen-Image 标注为 "Not validated"）。
- **MXFP4 的 DiT 部署**：上游 main 已支持（PR #5544 于 2026-08-25 合并）；等 release 或直接用 main。compressed-tensors MXFP4 在 diffusion 侧仍无路径。

## 6. auto-quant 框架能否覆盖（不含评测）

框架：`/home/kaokaolv/auto_agent/auto-quant`（aqa）。**骨架模型类别中立**（provider/harness/backend/scorer 都是 entry-point 插件），但**所有已实现的插件和 agent 领域知识都是文本 LLM 的**。只做量化（不管评测）需要动四处：

1. **命令生成（必须）**：`aqa/quantize/__init__.py:22-61` 的 builder 和 `QuantConfig`（`aqa/config.py:33-126`）是 LLM CLI 词汇。需加 diffusion 分支：允许 `guidance_scale/num_inference_steps/generator_seed/dataset(coco2014|tsv)` 参数，禁止 `--quant_lm_head` 等 LLM-only 参数；`config.py:67-79` 的 algorithm 白名单和 `:113-120` 的校验也要对 diffusion 放宽（diffusion 下 `--algorithm` 只接受 svdquant/auto_round/rtn，无 awq/hadamard 语义——实际上 awq/hadamard 对 DiT 也未必适用，需验证）。
2. **模型分析（必须）**：`aqa/agent/optimize.py:198-240` 的 `analyze_model` 只会读 LLM 的 config + safetensors index；diffusion 仓通常**没有顶层 `model.safetensors.index.json`**（fallback 只试单文件，`optimize.py:189`）。需改为读 `model_index.json`（`_class_name`）+ `transformer/` 子目录的 index，识别 DiT 结构；`ModelAnalysis`（`optimize.py:136-152`）的 MoE/router/MLA 字段对 diffusion 无意义；`validate_command` 的 MoE 强校验（`optimize.py:529-535`）要按模型类别跳过。
3. **recipe 模板与知识库（必须）**：`RECIPE_DOC_TEMPLATE`（`optimize.py:41-85`）和 `aqa/agent/recipes/auto_round_recipe.md` 全是 LLM 案例，需要 diffusion 版本（本文档可作底稿）；scheme YAML 需要长校准数据字段。
4. **fix-and-patch（少量）**：机制模型无关，但 patch 跟踪硬编码 `auto_round / vllm / model_code`（`skills/fix-and-patch/SKILL.md:28`），diffusion 场景要加 `diffusers`；fix prompt 的 COMPONENT 枚举与 playbook 需补 diffusion 案例（block 注册缺失、CUDA OOM at calibration、pipeline 加载失败等）。

**结论**：量化侧覆盖是"加一套 diffusion 分支"，不动骨架；工作量集中在命令构建 + 模型分析两个文件，以及 agent 知识库。评测侧（图像质量 harness/backend/gate）按你的要求本次不展开，但注意 evaluate skill 的 lm_eval/accuracy-gate 全链路对文生图不适用，将来要新写。

## 7. 试跑 checklist（任何新 diffusion 模型）

1. 读 `model_index.json` 的 `_class_name` → 确认 pipeline 类在**本机 diffusers 版本**里存在（Flux2Pipeline 需要较新 diffusers）。
2. 确认 `pipe.transformer`（或 `unet`）存在；打印 transformer 的 block 类名，对照 `block_runner.py:73-74` 的注册表。
3. 确认显存：校准 = 完整 pipeline 推理 × nsamples × steps，权重全量驻留（text encoder 虽不量化但参与前向）。
4. 先 `iters=1 nsamples=1 num_inference_steps=2` 冒烟，再放大。
5. 明确产物用途：fake = 进程内精度对比；**W4A16 + `auto_round` 格式 = vLLM-Omni 可部署**（第 5 节，前提 vllm-omni 支持该架构）；SVDQuant + Nunchaku 仅 FLUX.1。

## 8. 已知限制汇总（代码/文档明示）

- 只量化 transformer/unet；text encoder、VAE 不动（`README.md:11`）。
- fake 格式产物不支持重新加载（`README.md:12`）——要部署用 `auto_round` 格式走 vLLM-Omni（W4A16 已验证）。
- SVDQuant 端到端仅验证 FLUX.1-dev；nunchaku 导出拒绝非 FLUX adapter（`svdquant_nunchaku.py:167-171`）。
- diffusion 平台仅 HF（`utils/model.py:885-888`）。
- 双 transformer（Wan）时 `num_inference_steps<2` 强制提到 2（`diffusion_mixin.py:400-407`）。
- vLLM-Omni 侧 AutoRound MXFP4 已合入上游 main（PR #5544，2026-08-25）；W4A16/MXFP8 可用（第 5 节）。本地快照 `51b7565f` 早于合并点，用上游 main 或后续 release。
