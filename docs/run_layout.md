# Run 产物落盘结构与内容（run layout）

一次 `aqa run` / `aqa sweep` 的产物目录是**显式契约**，机器可读定义在
[`aqa/report/layout.yaml`](../aqa/report/layout.yaml)（single source of truth）。
用 `aqa inspect <run_dir>` 校验任何 run 目录是否符合契约（缺失项会直接列出）。

## 目录树

```
<run_dir>/
├── aqa_run.json              # 运行状态机（running/done/failed）                        [PLAN]
├── analysis.json             # 模型身份 + 计划量化范围（config.json 骨架，不下权重）     [ANALYZE] †
├── recipe_kb.json            # Recipe KB 热启动审计：query(含 current_stack) + 命中(含 trust/staleness)/null [ANALYZE] ¶
├── env_lock.json             # 冻结的软硬件栈（python/platform/包版本/accelerator）      [ENV] †
├── quant_plan.json           # 量化方案机读版：scheme/method/algorithm/iters + 冻结命令  [QUANT] †‡
├── .quant_command.json       # 实际执行的 auto-round 命令及来源（builder|agent-optimize）[QUANT]
├── quantized/                # 量化 checkpoint（权重；dataset sink 排除）               [QUANT]
│   └── quantize.log          # auto-round 全程日志
├── baseline/                 # bf16 基线（eval.baseline: true 时才有）                   [BASELINE]
│   └── accuracy.json
├── baseline_thinking_off/    # 基线的 thinking=off（双模式+baseline 时才有）            [BASELINE]
├── eval/                     # 评估产物（单模式或 thinking=on）                         [EVAL]
│   ├── accuracy.json         # {mean_score, results{task:{score,metric,num_samples}}, commands}
│   └── <task_group>/         # lm_eval 输出（results json + samples + .eval.log）
├── eval_thinking_off/        # thinking=off 模式（双模式时才有），结构同 eval/          [EVAL]
├── eval_report.json          # source(bf16) vs quant 的 mean_score + 逐 benchmark gap   [EVAL] †§
├── sweep_report.json         # aqa sweep 多候选对比（sweep 时才有）                     [EVAL]
├── gate.json                 # GO/NO-GO + 理由（thinking 双模式含 modes 明细）          [GATE]
├── run_manifest.json         # 复现清单：config + 环境版本 + 实际执行命令 + 分数        [CLOSE]
├── recipe.yaml               # INC-Recipes 兼容的 model-recipe（schema 1.0 + aqa 扩展）  [CLOSE]
├── <model_id>.yaml           # INC-strict recipe（无 aqa 扩展；可直接拷到 _data/recipes/）[CLOSE]
├── model_card.md             # 量化模型卡（有 eval 即写；无 signoff 为 draft；发布时作 README）[CLOSE]
├── reproduction_report.json  # 复现三面校验报告（仅 aqa reproduce）                      [VERIFY]
├── <artifact_name>/          # local_fs sink 拷贝（含 run_meta.json）                    [CLOSE]
├── logs/                     # 执行器日志 executor_*.log                                [跨阶段]
│   ├── agent_optimize/       # optimize agent 推荐过程
│   └── agent_fixes/<phase>/  # fix-loop 每轮修复记录
├── patches/<phase>/          # fix-loop 产出的可用补丁（owner 复查落点）                [跨阶段]
├── patch_snapshot/           # patch.capture 时工作区改动快照                           [跨阶段]
├── .aqa_local_docker/        # local docker staged 工件（调试，不上传）
├── .aqa_local_venv/          # local venv staged 工件（调试，不上传）
└── session/                  # P1+ 会话状态机（state.json/events.jsonl/manifest.json）
```

`[PHASE]` 标注产物归属的 Phase（RFC 0001 §4 的**阶段产物契约**）。带 † 的四个
`*.json` 是 P2 新增的**机读阶段产物**，只在走了 session 状态机的 run（存在
`session/`）里强制：
- `†` = `when: session`（`analysis.json` / `env_lock.json` 每个 session run 都有）
- `‡` = `when: session_quantize`（`quant_plan.json` 仅量化 run）
- `§` = `when: session_done`（`eval_report.json` run 正常收尾才有）
- `¶` = `when: optional`（`recipe_kb.json` 仅 `recipe_kb.enabled` 的量化 run 写：ANALYZE 读 KB 审计 query+命中，缺失不判 INCOMPLETE。命中带 `authority`/`confidence`/`stale`(+`stale_tools`)/`sessions` 可信度与时效信号；`current_stack` 与配方 pin 时的 `stack_fingerprint` 比对判定 stale。仅 `agent.optimize` 时命中才注入 agent-optimize 作起点。CLOSE 始终回写：GO→`recipes/quant/auto/<slug>.json` 验证配方(schemes+trust+session 行)；非 GO→同一行追加 negative knowledge(failures+session 行，不写 schemes))

`aqa inspect` 现在按 Phase 汇总（每个阶段 present/MISSING 计数），一眼看出
run 卡在哪个阶段、缺哪个产物。

CLOSE 同时写 `<model_id>.yaml`：这是 `to_inc_strict()` 投影后的 **INC-strict**
版本（不含 `quantization_details` / `provenance`），文件名等于 `model.id`，可直接
复制到 `INC-Quantization-Recipes/_data/recipes/` 并用
`bundle exec ruby tools/recipe_data.rb validate` 校验；`recipe.yaml` 保留 aqa
扩展供本仓审计。

条件性条目（`when`）的存在性由 run 配置决定：量化 run 必有 `quantized/` +
`.quant_command.json`；`eval.baseline: true` 才有 `baseline/`；thinking 双模式才有
`*_thinking_off/`；`agent.fix_loop: true` 才有 `patches/`；sweep 才有
`sweep_report.json`。判定规则见 layout.yaml 底部注释。

## 上传范围

- **dataset sink**（`hf_dataset`）：上传整个 run 目录但**结构性排除权重**
  （`quantized/` 与全部权重后缀）并做文本脱敏（`aqa/report/scrub.py`）；
  layout.yaml 中 `sink: false` 的条目（权重、staged 工件）不在其内。
- **local_fs sink** 镜像 `<artifact_name>/` 时会排除镜像目录本身；同一 run 目录
  重跑或刷新镜像不会把旧镜像递归复制成嵌套目录。
- **量化模型权重**默认不外传。量化 run 只要产出了评估精度，CLOSE 就会写
  `model_card.md`：无 `provider.upload`/signoff 时写 draft（`base_repo_id` 和
  signoff 用模板占位符，评估 table 按实际 task 全量写入）；发布时把
  `Signed-off-by` 换成用户当次确认的 `upload.signoff`，再原样作为 model repo 的
  `README.md` 上传。缺 sign-off 或仍是占位身份时跳过上传。

## 外部 agent（owner）的约定

owner 驱动 aqa 时，自己补充的产物按同一契约组织，保证 sink 能一并带走：

- 分析文档 / 决策记录 → 放 run 目录顶层（如 `<Model>_quantization_analysis.md`）
  或 `logs/` 下；dataset sink 会脱敏上传
- owner 的 fix/patch → 放 `patches/<phase>/`，与 aqa 自产补丁同构
- 运行命令与结论 → 记入 run 目录的文档，或依赖 `run_manifest.json` 已有的命令记录
- 验证一个 run 是否完整：一律 `aqa inspect <run_dir>`，不要凭记忆核对
