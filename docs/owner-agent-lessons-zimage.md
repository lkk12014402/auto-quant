# aqa Owner Agent — 实战经验沉淀（Z-Image-Turbo MXFP8/MXFP4 任务）

> 本文件沉淀本次 session 作为 aqa owner agent 的要点、遗漏点与规范。
> 使用 /models/auto-quant 前必读 `AGENTS.md`；本文件是它的补充（实证教训）。

## 0. 环境要点（本机特有）

- **session 位置**：本机 `HOME=/models/tools/copilot_home`，所以 copilot session 在
  `/models/tools/copilot_home/.copilot/session-state/<id>/events.jsonl`，
  **不是** `/root/.copilot/`。找 owner session 时用 `echo $HOME` 确认。
- **venv**：`/models/.venv-zimage`（vllm 0.28.0 + vllm_omni 0.28.0 + auto_round 0.15.0）。
- **代理**：`export https_proxy=http://proxy.ims.intel.com:911 && export http_proxy=...`
- **GPU**：用 `CUDA_VISIBLE_DEVICES=3`（GPU3 空闲）。
- **HF token**：`LOCAL_RESULTS_HF_TOKEN`（写 dataset `lvkaokao/lb_local` + model repo
  `INCModel3`/`INCModel` 等）；`HF_TOKEN` 用于下载。

## 1. 回传规范（本次最大的遗漏点）

### 1.1 量化模型默认不上传，必须配 `provider.upload`

`hf_dataset` sink 默认**排除权重**（`quantized/` 目录 + `.safetensors`）。要发布量化模型，
必须在 run yaml 配 `provider.upload`：

```yaml
provider:
  type: quantize
  upload:
    repo: INCModel3            # HF org（token 有写权限的 org）
    repo_name: Z-Image-Turbo-MXFP8-RTN-AutoRound
    token_env: LOCAL_RESULTS_HF_TOKEN
    model_card: true
```

- 不配 = 量化模型只留本机，不发布。
- **org 选择**：token 属于 `lvkaokao`，可写 `Intel`/`INCModel`/`INCModel2`/`INCModel3`/
  `INCModelSharing`。Z-Image 系列已发布在 `INCModel`（W4A16）与 `INCModel3`（MXFP8/MXFP4）。
- **model card 坑**：`HfModelSink` 自动生成的 model card 用 `pipeline_tag: text-generation`
  且 `base_model` 是本地路径（`/models/...`），对 diffusion 模型会 400 校验失败。
  解决：手动写 README.md（`pipeline_tag: text-to-image` + 合法 HF base_model），
  或 `model_card: false` 后手动补。

### 1.2 日志默认不在上传的 run 目录里

`executor_venv.log`（含量化+推理日志）在 run 的**父级** `logs/` 目录，不在 run 输出目录
（`<ts>/<ts>/`）内，所以 sink 不会带上。要回传日志：

```bash
mkdir -p <run_dir>/logs
cp <run_parent>/logs/executor_venv.log <run_dir>/logs/
cp <run_dir>/quantized/quantize.log <run_dir>/logs/
```

### 1.3 OWNER_JOURNAL.md 要放进 run 目录

owner journal 若放在任务级目录（如 `results/zimage-turbo/OWNER_JOURNAL.md`）不会被上传。
**约定：放进 run 输出目录顶层**（qwen3 系列就是这么做的），sink 自然带走。

### 1.4 owner session 轨迹要手动沉淀

aqa 自动沉淀 fix-loop/optimize 的 session，但 **owner 自己的 session 要手动**（见
`skills/session-sediment/SKILL.md`）：

```bash
mkdir -p <run_dir>/sessions
cp $HOME/.copilot/session-state/<id>/events.jsonl <run_dir>/sessions/copilot_<id>.jsonl
# 或离线转 ATIF 验证：
python -c "from aqa.agent.adapters import convert; from pathlib import Path; \
  a=convert(Path('<events.jsonl>'), backend='copilot'); print(len(a['steps']))"
```

### 1.5 org 推导 bug（已修）

`RunMeta.org` 从 `meta.model` 推导。本地路径 `/models/Z-Image-Turbo` → `split("/",1)[0]`=空
→ `_safe_name` → `"unknown"`，导致上传到 `results/unknown/...`。
**修复**：`OutputConfig` 加 `report_model` 字段；`RunMeta` 加 `org_override`；pipeline 用
`_report_org(run_cfg)` 从 `report_model` 取 org。yaml 里配 `report_model: Tongyi-MAI/Z-Image-Turbo`
即可让结果上传到 `results/Tongyi-MAI/...`。

## 2. MXFP8 量化 bug（本次核心排错）

**现象**：MXFP8 量化模型 geneval 0.0005、drawbench clip 18（baseline 31.77），而 MXFP4 正常。

**根因**：`OmniINCConfig.get_quant_method`（vllm_omni/quantization/inc_config.py）的 MXFP8
分支对所有 `LinearBase` 层无条件返回 `IncMxfp8OfflineLinearMethod`，**没检查
`block_name_to_quantize`**，导致 text encoder（Qwen3ForCausalLM）也被量化成 MXFP8，
文本嵌入 cap_feats 错误 → 整个生成崩溃。MXFP4 走父类逻辑会检查 block 列表，所以正常。

**修复**：MXFP8 分支复用父类 `config_parser.resolve(layer, prefix).quantized` 判断是否量化。
Patch：`/models/auto-quant/patches/fix_mxfp8_text_encoder_quantization.patch`。

**排错方法论**（可复用）：
1. 先验证 checkpoint 反量化正确（权重/scale 与原始 bf16 对比，误差应 ~2%）。
2. 从 model_runner 直接检查**加载后**的权重/scale（注意 MXFP8 是 [K,N] 布局 + swizzle scale，
   反量化要 un-swizzle）。
3. 逐层对比加载权重 vs checkpoint（136/136 层全对 → 加载没问题）。
4. 对比量化 vs bf16 的逐 block 输出，定位发散点。
5. **对比 text encoder 的 quant_method**：bf16/mxfp4 是 `UnquantizedLinearMethod`，
   mxfp8 是 `IncMxfp8OfflineLinearMethod` → 定位到 text encoder 被误量化。

## 3. 其它要点

- **forward 脚本正确性**：直接调 `tr.forward(x,t,cap_feats)` 与 `omni.generate` 第一步输出
  完全一致（max_abs 0.0），可放心用直接 forward 做逐层对比。
- **同进程不能加载两个 Omni 模型**（model parallel state 冲突），对比不同模型要分进程。
- **diffusion 纪律**：本地目录加载、`--disable_opt_rtn --disable_model_free`、
  `--ignore_layers adaLN_modulation`、coco2014 校准、8 步、guidance 0.0。
- **结果核对**：MXFP8 修复后 geneval 0.760（baseline 0.757）、clip 31.79（31.73），
  与 baseline 几乎一致；MXFP4 geneval 0.741。两者都可用。