# Run 产物落盘结构与内容（run layout）

一次 `aqa run` / `aqa sweep` 的产物目录是**显式契约**，机器可读定义在
[`aqa/report/layout.yaml`](../aqa/report/layout.yaml)（single source of truth）。
用 `aqa inspect <run_dir>` 校验任何 run 目录是否符合契约（缺失项会直接列出）。

## 目录树

```
<run_dir>/
├── aqa_run.json              # 运行状态机（running/done/failed）
├── run_manifest.json         # 复现清单：config + 环境版本 + 实际执行命令 + 分数
├── .quant_command.json       # 实际执行的 auto-round 命令及来源（builder|agent-optimize）
├── gate.json                 # GO/NO-GO + 理由（thinking 双模式含 modes 明细）
├── quantized/                # 量化 checkpoint（权重；dataset sink 排除）
│   └── quantize.log          # auto-round 全程日志
├── eval/                     # 评估产物（单模式或 thinking=on）
│   ├── accuracy.json         # {mean_score, results{task:{score,metric,num_samples}}, commands}
│   └── <task_group>/         # lm_eval 输出（results json + samples + .eval.log）
├── eval_thinking_off/        # thinking=off 模式（双模式时才有），结构同 eval/
├── baseline/                 # bf16 基线（eval.baseline: true 时才有）
│   └── accuracy.json
├── baseline_thinking_off/    # 基线的 thinking=off（双模式+baseline 时才有）
├── logs/                     # 执行器日志 executor_*.log
│   ├── agent_optimize/       # optimize agent 推荐过程
│   └── agent_fixes/<phase>/  # fix-loop 每轮修复记录
├── patches/<phase>/          # fix-loop 产出的可用补丁（owner 复查落点）
├── patch_snapshot/           # patch.capture 时工作区改动快照
├── <artifact_name>/          # local_fs sink 拷贝（含 run_meta.json）
├── sweep_report.json         # aqa sweep 多候选对比（sweep 时才有）
├── .aqa_local_docker/        # local docker staged 工件（调试，不上传）
└── .aqa_local_venv/          # local venv staged 工件（调试，不上传）
```

条件性条目（`when`）的存在性由 run 配置决定：量化 run 必有 `quantized/` +
`.quant_command.json`；`eval.baseline: true` 才有 `baseline/`；thinking 双模式才有
`*_thinking_off/`；`agent.fix_loop: true` 才有 `patches/`；sweep 才有
`sweep_report.json`。判定规则见 layout.yaml 底部注释。

## 上传范围

- **dataset sink**（`hf_dataset`）：上传整个 run 目录但**结构性排除权重**
  （`quantized/` 与全部权重后缀）并做文本脱敏（`aqa/report/scrub.py`）；
  layout.yaml 中 `sink: false` 的条目（权重、staged 工件）不在其内。
- **量化模型权重**默认不外传；需要时配 `provider.upload`（HF model repo，
  自动生成模型卡）。

## 外部 agent（owner）的约定

owner 驱动 aqa 时，自己补充的产物按同一契约组织，保证 sink 能一并带走：

- 分析文档 / 决策记录 → 放 run 目录顶层（如 `<Model>_quantization_analysis.md`）
  或 `logs/` 下；dataset sink 会脱敏上传
- owner 的 fix/patch → 放 `patches/<phase>/`，与 aqa 自产补丁同构
- 运行命令与结论 → 记入 run 目录的文档，或依赖 `run_manifest.json` 已有的命令记录
- 验证一个 run 是否完整：一律 `aqa inspect <run_dir>`，不要凭记忆核对
