# Qwen 系列 recipe 回刷 — owner agent 任务书（公共协议）

本目录每个文件是一个模型的完整 owner 任务书（可单独粘贴给 kimi cli / copilot cli）。
共性约定在此统一，单模型文件只写差异。diffusion 模型（flux2-*/z-image-turbo）的
任务书在评估口径上与 LLM 不同（DrawBench 三指标 + GenEval，无 v2/v3），且量化
纪律不同（见各任务书与 aqa `examples/run_diffusion_*.yaml`）。

## 适用范围与纪律

- owner 角色与铁律：先读 aqa 仓库的 `AGENTS.md`（任务书里以 `AQA_REPO` 指代，
  执行前先确认它的实际路径）。先 yaml 后运行、全程留痕、不与他人抢 GPU、
  把关 fix-loop、偏离要可复现。
- **每个变体一个 aqa run**（各自独立 run 目录，不要 sweep——要的是逐变体
  完整 run 产物与回传）。

### ignore_layers 纪律（重要，别图省事）

- **该 scheme 已有 recipe 页** → 沿用页面里的 Exclude layers（它们是验证过的）。
- **该 scheme 没有页面**（新建页）→ **禁止照抄同模型其它 scheme 的清单**。
  流程：owner 用 aqa 的免下载结构分析（config + safetensors index，见
  AGENTS.md §3.1）先出逐模块精度表初稿，再跑**多档 ignore 探索**（保守档 /
  激进档，可借助 `aqa sweep` 的自定义候选档 `recipe_sweep_candidates`），
  由精度对比选定入页面的最终清单。
- 分析先验（aqa `skills/quantize/SKILL.md` 的决策表里有完整版，agent 读它）：
  MoE 路由专家是量化主力（mxfp4）；非专家 Linear 可以 mxfp8；router/gate
  （`mlp.gate` 等）必须保高精度；embedding/lm_head/norm/conv 不保量化；
  小 N 层（out<128）不能上 MXFP8（FlashInfer 限制）。
- 量化方案矩阵（每模型 6 个变体，NVFP4 不做）：

| 变体 | quant 配置 | 评估差异 |
|---|---|---|
| mxfp8 v1 | scheme MXFP8, method model_free | 默认 |
| mxfp8 v2 | scheme MXFP8, method autoround, iters 0, static_kv_fp8: true | eval 加 `kv_cache_dtype: fp8` |
| mxfp8 v3 | v2 + static_attention_fp8: true | 同 v2 |
| mxfp4 v1 | scheme MXFP4（tuning 还是 model_free 见单模型任务书） | 默认 |
| mxfp4 v2 | v1 校准路径 + static_kv_fp8 | eval 加 `kv_cache_dtype: fp8` |
| mxfp4 v3 | v2 + static_attention_fp8 | 同 v2 |

- 评估矩阵：`eval.benchmarks: [gsm8k, piqa, mmlu, hellaswag, aime25, gpqa_diamond]`
  （前四个 lm_eval、后两个 evalscope），evalscope 段固定 `repeats: 5`（avg@5）。
  `baseline: false`——BF16 参考行用现有 accuracy-result 页的数字（Ratio 照它算）。
  `thinking: auto`（thinking 模型自动双模式，如实记录两组数）。
- 回传：每个 yaml 必须带
  `output.sinks: [local_fs, hf_dataset(repo=${LOCAL_RESULTS_DATASET:-lvkaokao/lb_local}, token_env=LOCAL_RESULTS_HF_TOKEN)]`
  和 `agent.knowledge.repo` 同 repo；`aqa validate` 应无提醒。
- 收尾：run 目录 `sessions/` 放 owner 自己的 session；每档跑完
  `aqa inspect` 验产物；环境偏离用 `aqa log-action` 记录；补丁
  `aqa capture-patch` 捕获（值得上游的另行 `publish-patch`，先审）。
- recipe 回写：按 aqa 仓库 `skills/recipe-page/SKILL.md` 执行，目标仓库
  **`/models/INC-Quantization-Recipes`**（不在本机就先确认访问方式），
  **不 commit、不直推**，改动留工作区并输出改动清单。
- 量化模型命名：`<Model>-<Scheme>[-FP8KV][-FP8Attn]-CT-AutoRound[-RTN]`
  （aqa `report/names.py` 规范；model_free 与 iters=0 都算 RTN 档）。
- 每个跑不动的变体：如实记录失败原因进任务日志和页面 `_待补_`，禁止编造数字。

## 环境（/models 所在机器）

- 执行形态：`executor.type: local` + `env.type: venv`（`packages: auto`）；
  qwen3-8-flash-next 例外（day-0，用官方 docker 镜像路径，见单模型任务书）。
- 开始前 `source env.sh`（HF_TOKEN/LOCAL_RESULTS_*），`nvidia-smi` 选空闲卡；
  `num_gpus` 与 TP 按单模型任务书的资源表。
