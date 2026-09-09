# Owner 与 aqa 的接口契约（两种运行模式）

aqa 有两种合法运行模式，本文件定义两者与 aqa 的沟通/连接方式。

## 模式 A：owner agent 驱动（copilot / claude / kimi cli 等外部 agent）

owner 负责：架构判定、yaml 编写、运行监督、fix-loop/optimize 行为把关、
环境调整、结果核对、知识刷新。aqa 负责执行。铁律见 `AGENTS.md` §0。

**owner → aqa**（owner 怎么影响 run）：

| 通道 | 形式 | 说明 |
|---|---|---|
| yaml 配置 | `aqa run/sweep <yaml>` | 主通道：环境/量化/评估/落盘全部经此声明（先 yaml 后运行） |
| `owner_actions.jsonl` | `aqa log-action <run_dir> --kind env --text "..." [--command "..."]` | run 期间的环境变更/额外命令/决策记录；run 结束时并入 `run_manifest.json` 的 `owner_actions` 段，随 sink 上传 |
| `patches/owner/` | `aqa capture-patch <run_dir> --repo <git仓>` | owner 对依赖工作区的手改捕获为补丁（git diff HEAD；干净树跳过并提示），与 aqa 自产补丁同构随 sink 上传 |
| 补丁转 PR | `aqa publish-patch <patch文件> --repo-dir <fork本地仓> --title ... [--dry-run]` | 把补丁变成 fork 的真实 PR：同步 upstream→fork 基线 → 临时 worktree（不碰本地脏树）→ apply 校验 → commit → push → `gh pr create`。token 只从环境（GH_TOKEN/GITHUB_TOKEN）读 |
| 知识刷新 | `aqa knowledge push-references` | references 增量层（见 `docs/knowledge_refresh.md`） |

**aqa → owner**（owner 怎么审查 run）：

| 通道 | 位置 |
|---|---|
| 产物完整性 | `aqa inspect <run_dir>`（契约：`aqa/report/layout.yaml`） |
| 复现清单 | `run_manifest.json`（config + 环境版本 + **实际执行命令** + owner_actions） |
| fix-loop 行为 | `logs/agent_fixes/<phase>/`（会话日志 + root_cause + 补丁） |
| optimize 行为 | `logs/agent_optimize/`（recipe 文档 + 会话日志）；最终命令在 `.quant_command.json` |
| 远端 | HF dataset `results/<org>/<artifact>/run_<run_id>/` |

## 模式 B：纯 `aqa run`（无 owner agent）

`aqa run <yaml>` 完全自主：fix-loop 自愈、optimize 推荐、sink 回传，
**不强制任何外部 agent 在场**。此时：

- manifest 的 `owner_actions` 段自然缺省（`null`），不报错
- 把关由人事后完成：`aqa inspect` 看完整性，翻 `logs/agent_fixes/` 的
  session 看修复行为，对照 manifest 的 `commands.quantize_source`
  （`builder` = 确定性构建 / `agent-optimize` = agent 推荐）判断命令来源
- 人也可以随时补记动作（事后审计/复现备注）：

```bash
aqa log-action <run_dir> --kind decision --text "为何选 MXFP4 而非 W4A16：..."
```

- **无人模式的风险边界**：fix-loop 默认 `max_fix_attempts: 5` 内自决，
  修不通会以 `Quant Failed`/`Eval Failed` 状态回传——榜单上可见，
  不会静默。要更保守就 `agent.fix_loop: false`（纯确定性 builder，出错即停）。

## 补丁的决策与审核（owner 职责）

aqa 的 fix-loop **捕获即生效，没有审核闸**——补丁审核是 owner 的事后行为
（纯 `aqa run` 无人模式下由人事后完成）。协议：

1. **审什么**（全部在 run 目录里）：
   - `patches/<area>.patch`（累计净 diff）+ `patches/attempts/*.patch`（单次）
   - `patches/manifest.json` 的两个自检字段：`applies_clean`（反向 dry-run 能否
     干净回滚）与 `resolved`（该补丁是否真的让相位通过）
   - `logs/agent_fixes/<phase>/` 的会话与 root_cause——补丁的"为什么"
2. **怎么判**：diff 是否只动了该动的文件、根因是否有证据、
   `applies_clean=true` 而 `resolved=false` 的补丁是"没修对的改动"——重点看。
3. **怎么记录决策**（进 manifest 随 sink 上传）：

```bash
aqa log-action <run_dir> --kind decision \
  --text "接受 patches/quantize.patch：根因=X，证据=Y"
aqa log-action <run_dir> --kind decision \
  --text "否决并回滚 patches/evaluate.patch（误删了 shared_expert 量化）" \
  --command "git apply -R <run_dir>/patches/evaluate.patch"
```

4. **否决后怎么做**：`git apply -R` 回滚该补丁（applies_clean=true 保证可回滚）→
   修正 yaml/环境 → 重跑（知识库里已沉淀的错误签名会防止重蹈覆辙）。
5. 倾向"先审后放行"的场景：把 `agent.fix_loop` 关掉跑确定性 builder，
   出错即停、人工修——完全消除未审补丁。

> 注：`patch.review` 配置字段当前在代码里**没有消费者**（占位），
> 不要指望它产生 review.md；审核按上面协议走。

## 两种模式共用的纪律


- 任何环境/代码偏离都要可复现：yaml `env.packages` pin 或 `patches/` 补丁
- manifest 是审计真源：怀疑结果先看 `commands`（实际跑了什么）与
  `environment.packages`（版本），再看 owner_actions
