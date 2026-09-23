# <Model> <Scheme> 量化过程日志（journal）

> 模板使用：全程随手记，不是收尾补记。粒度参照 `docs/RUN_JOURNAL.md`。
> 每条记录三要素：**时间 + 命令/动作 + 依据或结果**。坑单独成节，
> 一条坑 = 现象 + 排查链 + 根因 + 修法 + 教训（这是第⑤步经验路由的输入）。
>
> 分工：本文件记**叙事与决策**（为什么这么做）；改环境/额外命令这类机器可读
> 动作同时跑 `aqa log-action <run_dir> --kind ...` 进 `owner_actions.jsonl`，
> 两份不重复。

> 任务：<一句话：模型 + scheme + method + 目标>
> 工作区：<路径>；run 目录：<output.dir>
> 开始：<YYYY-MM-DDThh:mmZ>

## 环境基线

- GPU：`<型号×数量，开始时空闲情况（nvidia-smi 实测）>`
- 环境：`<venv 路径 + 关键包版本（来自 setup_venv.sh 自检输出）>`
- 代理 / 特殊环境变量：

## 第①步：结构分析

- 结论摘要：`<架构一句话 + 量化范围一句话>`
- 产物：`analysis.md`（查阅清单见该文件 §0）
- 关键判断与依据：`<为什么这个 ignore 清单 / 为什么这个环境形态>`

## 第②步：环境

- 命令：`./setup_venv.sh`（自检输出摘要 / 踩到的系统级前置）
- requirements.txt 已冻结：`<YYYY-MM-DD>`；后续换依赖的重冻结记录追加在 §杂项

## 第③步：yaml 定稿与运行

- `aqa validate run.yaml` 结果：
- `aqa run run.yaml --dry-run` 计划要点：
- 正式运行：`<命令 + setsid/日志路径>`，run_id：`<>`

## 运行观察（按时间追加）

### <YYYY-MM-DD hh:mm> <阶段：baseline / quant / eval / gate>

- 动作 / 命令：
- 结果 / 日志路径：
- 下一步：

## 坑与修复（每条一节，编号①②③…）

### 坑①（<发现时间>，<状态：已修/绕过/未决>）<一句话标题>

- 现象：`<报错串 / 异常行为，附日志位置>`
- 排查链：`<每一步怎么排除的，实证命令>`
- 根因：
- 修法：`<yaml 改动 / patch 文件 / wheel；是否已验证>`
- 教训：`<可泛化的那条——第⑤步路由表候选>`

## 第⑤步：收尾记录

- sanity check：`<产物体积 vs 位宽预期；量化键 dtype 抽查结论>`
- gate：`<GO / NO-GO + 口径说明>`
- sink 回传核对：`<dataset 路径 / 文件数>`
- 产物清单核对：`aqa artifacts check <run_dir>` 输出摘要
