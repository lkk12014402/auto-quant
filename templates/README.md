# templates/ — 每个模型量化任务的产物骨架

本目录是 AGENTS.md「硬流程」的模板真源。**骨架 vs 实例**：本目录是骨架，
`cookbooks/`（实例分析/脚本）、`docs/RUN_JOURNAL.md`（实例日志）、
`templates/examples/model_card_gemma-4-31B.md`（合格 baseline model card，
gemma-4-31B-MXFP8-FP8KV）是填出来的样子。

**可移植纪律**：骨架文件（本目录顶层）不许出现任何本机绝对路径——本仓库会换
机器、会给别人用。实例文件（`examples/`）是历史实录，其中的路径属于产生它的
那台机器，照抄结构，不照抄路径。

## 使用契约

1. 每个新模型任务：把对应模板拷到该模型的工作目录，逐项替换 `<...>`/`{...}` 占位。
2. 填写纪律：章节结构不许删（不需要的章节写「本次无，原因」）；所有数字必须是
   **实测值**并注明测量方式/日期；拿不准的标 `TODO(owner)`，不许留空。
   **例外：`model_card.md`** 面向下载权重的用户，按其文件头的生成规则与禁写清单
   填写：章节固定、不写测量方式/审计证据（这些留在 `analysis.md` / `report.md`），
   不允许出现 `TODO` 或「本次无」；末行必须是 `Signed-off-by: Name <email>`，身份每次
   向用户询问确认（占位身份或复用实例卡的身份都会被 `aqa artifacts check` 拒绝）。
3. 产物归属：任务收尾时 7 个产物拷进 run 的 `output.dir` 顶层，随 sink 回传
   （文本自动脱敏，权重结构性排除）：

| 模板 | run 目录里的名字 | 产生时点 |
|---|---|---|
| `analysis.md` | `analysis.md` | 第①步（结构分析与量化方案） |
| `setup_venv.sh` | `setup_venv.sh` | 第②步（环境） |
| `requirements.txt` | `requirements.txt` | 第②步冻结，之后每次换依赖重冻结 |
| `run.yaml` | `run.yaml`（实际运行文件可另名，此为初始模板） | 第③步 |
| `journal.md` | `journal.md` | 全程随手记 |
| `model_card.md` | `model_card.md` | 第④步（仅发布权重时必填） |
| `report.md` | `report.md` | 第⑤步（回溯） |

4. 收尾校验：`aqa artifacts check <run_dir>` 做确定性检查（文件在不在、必备
   章节齐不齐、report 的 gate 结论与 gate.json 是否一致）。全过才算收尾完成。
