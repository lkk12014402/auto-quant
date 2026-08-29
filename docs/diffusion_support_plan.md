# aqa Diffusion 模型量化支持设计方案

> 日期：2026-08-25。状态：**M1+M2 已 GPU 实跑验收**：Z-Image-Turbo W4A16（CLIP 28.76→27.76，gate GO）与 FLUX.2-klein-4B MXFP8（CLIP 27.38→27.29，gate GO）全链路通过（auto-round RTN → vllm-omni → DrawBench CLIP，H200）。M2 模型面已落地（五个模型示例 + optimize DiT 分支 + omni_kwargs 透传 + 导出后自动修复，见 §3.2 注记）；auto-round 知识库（含 diffusion）已接入 agent。
> 依据仓库：`/home/kaokaolv/auto_agent/auto-quant`（本仓）、`/home/kaokaolv/auto_agent/auto-round` @ `bd9a13ce`、`/home/kaokaolv/auto_agent/vllm-omni` main @ `51b7565f`、`/home/kaokaolv/auto_agent/diffusion_eval`。

## 1. 背景与目标

aqa 当前只覆盖 text-generation LLM：auto-round CLI 量化 → lm_eval（hf/vllm）评估 → accuracy gate → agent fix-loop（`examples/` 下全部为文本 LLM 示例）。

**目标**：支持 diffusion 文生图模型（DiT：FLUX.1/FLUX.2 系、Z-Image、Wan2.2 等）的完整链路——

```
auto-round 量化（auto_round 格式，W4A16 / MXFP8 / MXFP4）
  → vllm-omni 推理
  → 图像质量评估（DrawBench + CLIP score 起步）
  → baseline 对比 gate + agent fix-loop（复用现有机制）
```

非目标（本阶段）：视频模型评估、FID/LPIPS 等"越低越好"指标、agent-optimize 的完整 DiT 知识库。

## 2. 现状分析（事实，带出处）

### 2.1 量化侧：链路已存在，aqa 只需编排

- auto-round CLI 原生支持 diffusion：`--dataset coco2014`（默认）、`--num_inference_steps`、`--guidance_scale`、`--generator_seed`（`auto_round/cli/parser.py:277-290`）；模型经 `model_index.json` 自动识别为 diffusion（`auto_round/utils/model.py:1290-1334`），只量化 `pipe.transformer`/`pipe.unet`，text encoder/VAE 不动（`auto_round/compressors/diffusion/README.md:11`）。
- aqa 的 `build_autoround_command` 是纯字段映射 + `extra_args` 透传（`aqa/quantize/__init__.py:22-61`），**确定性构建路径对 DiT 已可用**。
- 导出格式：DiT → vllm-omni 必须用 `--format auto_round`（vllm-omni 按 `quantization_config.quant_method="auto-round"` 自动检测，`vllm-omni/docs/user_guide/quantization/autoround.md`）。aqa `QuantConfig.export_format` 默认 `llm_compressor` 是 LLM 取值，run.yaml 里显式写 `auto_round` 即可。

### 2.2 推理侧：vllm-omni 的 AutoRound 支持矩阵

| scheme | vllm-omni main 状态 | 出处 |
|---|---|---|
| W4A16（INT4, auto_round:auto_gptq） | ✅ 已验证：FLUX.1-dev、Wan2.2、GLM-Image、Cosmos3、Qwen-Image | PR #1777（commit `c1d2dcc3`）；`vllm_omni/quantization/inc_config.py` |
| MXFP8（data_type=mx_fp, bits=8） | ✅ 代码支持（`IncMxfp8OfflineLinearMethod`） | PR #3782（commit `1903cee5`）；`inc_config.py:66-84,180-277` |
| MXFP4（mx_fp, bits=4） | ✅ 上游 main 已合入（2026-08-25）；本地快照早于合并点 | PR #5544（merged）；用上游 main 或后续 release |

checkpoint 要求：transformer 子目录 `config.json` 含 `quantization_config`（quant_method/bits/group_size/packing_format/**block_name_to_quantize**）。`block_name_to_quantize`（如 `"transformer_blocks,single_transformer_blocks"`）是否由 auto-round diffusion 导出自动写入，需首次实跑确认，缺失则后处理补写。

硬件：NVIDIA SM80+ / Intel XPU（autoround.md 硬件表）。

### 2.3 评估侧：diffusion_eval 的真相

`/home/kaokaolv/auto_agent/diffusion_eval` **不是评估框架**：只有两个脚本，无指标计算代码。

- `run_drawbench_bf16_local.py`：DrawBench（HF `sayakpaul/drawbench`，200 条 prompt，字段 `Prompts`/`Category`）生图脚本，diffusers 本地推理，产出 PNG + `metadata.csv`（idx/category/prompt/image_path/seed）。模型配置硬编码于 `MODEL_CONFIGS`（FLUX.2-dev/klein-4B/9B、Z-Image-Turbo）。**生图与打分天然两阶段分离**——替换推理后端只需改 `load_pipeline`/`generate_single` 两个函数。
- `infer_deepseek_ocr2_omnidocbench.py`：OCR VLM 推理，与文生图无关，不动。

指标需新接。第一指标选 **CLIP score**（越高越好，与现有 gate 兼容；实现用 `transformers.CLIPModel`，依赖最简）。vllm-omni 自带的 `compare_diffusion_trajectory_similarity`（PSNR/MAE/cosine 等像素级指标）可作后续补充。

vllm-omni 推理 API 参照 `vllm-omni/examples/offline_inference/text_to_image/text_to_image.py`（已支持 FLUX.2-dev/klein-4B/9B、Z-Image-Turbo、FLUX.1-dev）：`Omni(model=..., ...)` + `omni.generate(prompt, OmniDiffusionSamplingParams(...))`。

### 2.4 aqa 骨架：模型类别中立，实现全是 LLM

- 中立且**不用改**：provider 机制（`aqa/providers/`）、pipeline 主干（`aqa/pipeline.py:331-411`）、fix_loop 状态机、accuracy.json schema（`aqa/eval/registry.py:191-206`）、gate 机制（`aqa/report/__init__.py:27-78`）、sinks/manifest。
- 全部是 LLM 的：harness（lm_eval/native）、backend（hf/vllm）、benchmark recipes（15 个文本任务）、optimize prompt 模板。agent 知识库（`aqa/agent/recipes/*.md`）原本只有 LLM 的 MXFP4/MoE 蒸馏知识，**本次已补充** `auto_round_diffusion.md`（diffusion/VLM 蒸馏版）并在 `optimize.py` 按 `is_diffusion` 注入；完整版参考落在 `docs/autoround_full_reference.md` 与 `docs/autoround_diffusion_reference.md`。
- 插件先例：evalscope harness 是外部 pip 包（`aqa/eval/harness/__init__.py:5`、`aqa/config.py:217` dict passthrough）——diffusion harness 照此模式接入。
- 死字段（勿依赖）：`QuantConfig.recipe`（`config.py:39` 无人消费）；eval heal 预 pass 未接入 pipeline。

## 3. 修改清单

### 3.1 必须改（M1 范围）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `aqa/eval/harness/diffusion.py`（新） | `DiffusionHarness`：读 recipe 的 prompt 集 → vllm-omni 生图 → CLIP score → `BenchmarkResult`。**实现 `run_group`**：同组 recipe 共享一次模型加载（DiT 加载贵） |
| 2 | `aqa/backends/vllm_omni.py`（新） | `name="vllm_omni"`；协议要求的 lm_eval 方法留空（diffusion harness 不消费）；`env_overrides()` 给 omni 环境变量 |
| 3 | `aqa/config.py` | `EvalConfig` 加 `diffusion: DiffusionEvalOptions`（prompt dataset/steps/guidance/size/seed/metrics/limit），仿 `LmEvalOptions` |
| 4 | `aqa/eval/runner.py:22-41` | `_make_harness` 加 diffusion 分支，从 `cfg.diffusion` 取构造参数 |
| 5 | `aqa/registry.py` + `pyproject.toml` | 双轨注册 harness `diffusion` 与 backend `vllm_omni` |
| 6 | `recipes/benchmarks/drawbench_clip.yaml`（新） | `harness: diffusion`。裸任务名会 fallback lm_eval（`eval/registry.py:241`），diffusion benchmark **必须有 recipe 文件** |
| 7 | `aqa/executors/_remote.py:267` | backend 判断扩 `vllm_omni` → 容器装 vllm-omni + diffusers + 评分依赖 |
| 8 | `examples/run_diffusion_flux1.yaml`（新） | M1 示例：FLUX.1-dev + W4A16 + auto_round 格式 + DrawBench CLIP + baseline gate |
| 9 | `aqa/agent/prompt.py:34` + `aqa/agent/recipes/fix_playbook.md` | COMPONENT 枚举加 `vllm_omni`、`diffusers`；playbook 加 diffusion 段落（否则 fix-loop 分类失准） |

### 3.2 M2+（部分已落地）

- `aqa/agent/optimize.py`：**已全部落地**——`ModelAnalysis.is_diffusion` 检测（`model_index.json` 探测 + 读 `transformer/` 子目录 config/index）、`load_recipe_knowledge(include_diffusion=...)` 按模型类型注入 `auto_round_diffusion.md`、DiT 版 prompt 分支（`DIT_CLI_EXAMPLES` + DiT 硬要求 + intro）。examples 默认仍关 `agent.optimize`（确定性 builder），需要时打开。
- **M2 模型面已落地**（examples/）：`run_diffusion_zimage.yaml`（W4A16 RTN，8 步 guidance 0.0）、`run_diffusion_flux2_klein4b.yaml`（MXFP8 RTN）、`run_diffusion_flux2_klein9b.yaml`（W4A16 RTN）、`run_diffusion_flux2_dev.yaml`（W4A16 RTN，28 步 guidance 3.5）、`run_quant_deepseek_ocr2.yaml`（VLM 文本路径量化 + piqa 加载 smoke；OmniDocBench 评估为后续工作）。`DiffusionEvalOptions.omni_kwargs` 透传 `Omni(...)` 引擎旋钮（offload 等）。
- **重要事实（新确认）**：纯 RTN（`iters 0` + `--disable_opt_rtn`，即 `need_calib=False`）走 `_quantize_zero_shot()`，**完全不跑 pipeline 校准**（`diffusion_mixin.py:348-350`）——因此 Flux2 双输出 block 未注册（`Flux2TransformerBlock`/`Flux2SingleTransformerBlock`，见 `block_runner.py:71-78`）**只影响 iters>0 的校准路径**，RTN 不受影响。FLUX.2 要开校准调优需先在 auto-round 补 `register_diffusion_output` 注册。
- FID/LPIPS 等"越低越好"指标：`ScoringSpec` 加 `higher_is_better`，gate 按方向取反（`report/__init__.py:45`）。
- Wan2.2 视频评估（`text_to_video.py` 链路）、ImageReward 指标（已落地为 `aqa/eval/image_metrics.py`）、MXFP4（PR #5544 已合入上游 main，可用）。
- `aqa/report/manifest.py:34-37` `_TRACKED_PACKAGES` 加 vllm-omni/diffusers。

### 3.3 明确不用改

`aqa/providers/`、`aqa/quantize/__init__.py`、`aqa/pipeline.py` 主干与 fix-loop 包裹、`aqa/agent/fix_loop.py`、`aqa/eval/harness/lm_eval.py`/`native.py`、`aqa/report/stats.py`、registry 机制本体、`aqa/sandbox/`、`aqa/swebench/` 等。

## 4. M1 设计细节

### 4.1 数据流

```
run_diffusion_flux1.yaml
  → baseline eval（bf16 FLUX.1-dev → DrawBench 生图 → CLIP score）
  → quantize（auto-round --scheme W4A16 --format auto_round --dataset coco2014
              --iters 0 --disable_opt_rtn --num_inference_steps 28 --guidance_scale 3.5）
  → evaluate（量化 checkpoint → 同 prompt 集 → CLIP score）
  → gate（max_accuracy_drop vs baseline）
```

### 4.2 DiffusionHarness 契约

- `name = "diffusion"`；`available()`：检查 `vllm_omni` 可导入。
- 构造参数来自 `EvalConfig.diffusion`：`num_inference_steps / guidance_scale / height / width / seed / metrics / scorer_model（CLIP 型号）`。
- `run(recipe, model, backend, out_dir)`：
  1. 从 recipe 的 `dataset`（HF uri 或本地 jsonl/csv）读 prompt 列表 + `limit` 截断；
  2. `Omni(model=model)` 加载一次，逐 prompt 生图存 `out_dir/images/{idx:04d}.png` + `metadata.csv`；
  3. CLIP score 逐图打分（`transformers.CLIPModel`），`score = mean × 100`（原始 cosine × 2.5 的惯例放大可选，需在同一实现内自洽）；
  4. 返回 `BenchmarkResult(name, score, metric="clip_score", num_samples, per_sample=[...], command=<实际执行方式>)`。
- `group_key(recipe)`：按生成参数分组；`run_group(...)` 同组共享 Omni 实例。

### 4.3 recipe 示例（`recipes/benchmarks/drawbench_clip.yaml`）

```yaml
name: drawbench_clip
harness: diffusion
type: generative
dataset: sayakpaul/drawbench     # HF datasets；字段 Prompts/Category
prompt_field: Prompts
scoring:
  metric: clip_score
  prefer: [clip_score]
```

### 4.4 已知风险

1. ~~MXFP4 在 vllm-omni 本地 main 不可加载（PR #5544 未合）~~ PR #5544 已合入上游 main（2026-08-25）→ MXFP4 可用（用上游 main 或后续 release）；本地快照早于合并点时 M1/M2 用 W4A16/MXFP8。
2. ~~`block_name_to_quantize` 自动写入待实跑确认~~ **已确认**：Z-Image 导出自动写入（`noise_refiner,context_refiner,layers`）。
   - 实跑新发现并解决（全部沉淀进 `fix_playbook.md` diffusion 段 + `auto_round_diffusion.md`）：① `iters 0 + disable_opt_rtn` 自动路由 model_free → diffusion 空量化，必加 `--disable_model_free`；② diffusion 常规加载只认本地目录；③ 导出损坏 text encoder 权重名（丢 `model.` 前缀）→ aqa `repair_diffusion_export` 自动修复；④ vllm-omni 写死全精度的小层（Z-Image `adaLN_modulation`）不能被量化 → `--ignore_layers adaLN_modulation`；⑤ 同进程二次 Omni 加载需 `destroy_distributed_env()`（harness `_release_omni` 已修）。
3. 校准代价：diffusion 校准 = `nsamples × num_inference_steps` 次完整去噪（`auto_round/calibration/diffusion.py:98-213`），FLUX.1-dev 建议 iters=0 起步。
4. `batch_size` 在 diffusion 下被强制为 1（`auto_round/compressors/diffusion_mixin.py:86-113`）。
5. 评分模型（CLIP）首次运行需下载权重（~600MB）。
6. **Flux2 校准缺口**：iters>0 需先给 auto-round 补 `Flux2TransformerBlock`/`Flux2SingleTransformerBlock` 的 `register_diffusion_output` 注册；RTN 路径不受影响（§3.2 已确认）。
7. M2 新示例（Z-Image/FLUX.2 系）尚未 GPU 实跑；vllm-omni 的 auto_round W4A16 验证记录目前只有 FLUX.1-dev/Wan2.2/GLM-Image/Cosmos3，Z-Image 与 FLUX.2 的 auto_round checkpoint 加载需首跑确认。

## 5. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M1**（本次） | harness/backend/recipe/example + 设计文档；FLUX.1-dev + W4A16 + DrawBench CLIP 全链路 | `aqa run examples/run_diffusion_flux1.yaml` 产出 accuracy.json + gate.json；量化产物被 vllm-omni 直接加载 |
| **M2**（代码已落地，待 GPU 实跑） | Z-Image-Turbo / FLUX.2-klein-4B/9B / FLUX.2-dev / DeepSeek-OCR-2 示例；MXFP8；optimize.py DiT 分支（is_diffusion + 知识注入 + DiT prompt）；`omni_kwargs` 透传 | 新模型 run.yaml 在 GPU 机器跑通出 accuracy.json + gate.json；`agent.optimize: true` 时产出合法 DiT 命令（单测已覆盖 prompt 分支） |
| M3 | MXFP4（PR #5544 已合入上游 main，可实跑验证）；ImageReward/FID（gate 方向标志）；Wan2.2 视频 | 对应链路跑通 |

## 6. GenEval 对齐验证的终局记录（2026-08-26）

目标：aqa 的 GenEval 实现 vs Z-Image 官方发布值（[Table 9](https://arxiv.org/html/2511.22699v1)，Z-Image-Turbo overall **0.82**）。

**最终成绩（Z-Image-Turbo bf16，553×4，vllm-omni 生图 + transformers Mask2Former 判分）：overall 0.7579**（single 1.00 / two 0.90 / counting 0.71 / colors 0.86 / position 0.50 / color_attr 0.58）。

**修复轨迹**（每个都是真实 bug，全部带回归测试进仓）：0.19（初版）→ 0.62（修掩码窃取：HF argmax 合并掩码会抢走高分段的全部像素 → 空掩码被丢弃 → 漏检；改 `return_binary_maps`）→ 0.75（修类名别名：HF/mmdet 命名 motorbike/tvmonitor/sofa/pottedplant/... vs GenEval 的 COCO 官方名 motorcycle/tv/couch/...，9 项别名表）→ 0.7579（修分数语义：HF segment score = class_prob × mask_quality 系统性偏低，改为按类概率过滤对齐 mmdet 口径；另修 transformers 5.x 的 `get_*_features` 返回对象不再带 `.norm`、PIL composite 的掩码 dtype）。

**残余 -0.062 的归因**：判分侧已穷尽（position/color_attr 对检测器改动免疫），证据指向**生成侧协议**——Z-Image 官方评测的推理配置（prompt enhancer 是否启用、seed 策略）未公开。single_object 双方 1.00 完全一致是协议保真的锚点。

**结论与使用纪律**：GenEval 在 aqa 投入量化 A/B 回归使用（同路径相对比较有效）；绝对值对外引用时须注明「本仓实现 vs 官方 -0.062，归因为生成侧协议未公开」。要彻底追平需官方评测配置；mmdet 官方路径交叉验证留作可选（mmcv 在新 torch 上构建成本高）。
