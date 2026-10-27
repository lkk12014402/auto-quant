# Knowledge 与 Session 的沉淀机制

真源：`aqa/agent/knowledge.py`（KnowledgeStore）、`aqa/agent/lessons.py`、
`aqa/agent/knowledge_extract.py`、`aqa/agent/trajectories.py`、
`aqa/agent/adapters/`、`aqa/cli.py` 的 `traces` 命令组。
结构契约：`aqa/agent/knowledge_layout.yaml`。

## 1. Knowledge 沉淀的三条路径

### 1.1 run 中实时（自动）

fix-loop 每轮尝试产出结构化记录（`COMPONENT/ERROR_CLASS/ROOT_CAUSE/EVIDENCE/
VERDICT/FIX_TIER/FIX_PLAN`）。修复成功后：

- **lesson**：按错误签名（`extract_signature`，错误文本归一化）写入
  `lessons/*.jsonl`——key 是错误不是模型，天然跨模型/跨机器复用
- **可回放补丁**：归档到 `patches/<phase>/<签名哈希>/`（patch.diff + manifest）；
  下次同签名错误**先回放补丁再调 agent**（`knowledge.replay: true`）
- run 结束 `push_up` 回 HF dataset（`agent.knowledge.repo` 配置时），
  run 开始 `sync_down` 拉回；统计（replay_hit/hit_rate）在结束日志里打印

### 1.2 轨迹蒸馏（离线，事后）

```bash
aqa traces distill --path <trajectories.jsonl 或 run 目录> [--knowledge-root <dir>]
```

`knowledge_extract` 对 ATIF 轨迹做规则抽取：观测里识别 error、命令里识别
修复动作、进度变化判定成功，产出 error→fix 条目写回知识库。
**规则抽取是保底下限**——接 agent summarizer 可进一步提炼 root_cause。

### 1.3 references 人工维护

owner 维护的参考知识（auto-round recipe 等）：`aqa knowledge push-references`
（整树替换或 `--file` 单文件增量），逐文件 `auto_round_ref` 记新鲜度。
刷新流程见 `docs/knowledge_refresh.md`。

## 2. Session 沉淀的两层

### 2.1 run 内审计（自动）

| 来源 | 落点 |
|---|---|
| fix-loop 每轮 | `logs/agent_fixes/<phase>/<stem>.log` + `<stem>.events.jsonl`（结构化事件流，截断也可审计） |
| optimize | `logs/agent_optimize/optimize.log` + `quantization_recipe.md` |

session_id 形如 `fix_<phase>_<hash8>`；**tier 间会话不共享**（换后端即新会话），
跨 tier 的信息桥由 aqa 传结构化字段（历史假设压缩行）。

### 2.2 离线轨迹数据化（手动/按需）

```bash
aqa traces convert --path <session日志> [--backend kimi|claude|codex|copilot|openclaw] --out <dir>
aqa traces export  --path <dir> --view sft|rl [--sharegpt] --out <rows>.jsonl
aqa traces publish --path <dir>... --stage-only --stage-dir archive   # 去 --stage-only 加 --repo 推 HF
aqa traces distill --path <dir>                                       # 回流知识库
```

session 既是审计材料也是训练数据：convert → ATIF 轨迹；export → SFT/RL/ShareGPT
数据集行；publish → HF dataset 分层目录；distill → 知识。**全链路自动脱敏**
（适配器收口处执行，规则集 `aqa/report/scrub.py`）。

### 2.3 run 收尾自动沉淀（`agent.knowledge.sediment_sessions`，默认开）

run 结束（**成功和失败都执行**——失败的 session 最有教学价值）时，
pipeline 自动把本 run 的 session 转成 ATIF 并回传：

- **aqa 内部**：`logs/agent_fixes/**/*.events.jsonl`、`logs/agent_optimize/` 下的
  结构化会话 → 自动转换
- **owner 的 session**：aqa 碰不到——owner 把自己的 session 文件放进
  `<run_dir>/sessions/`（收尾 skill：`skills/session-sediment/SKILL.md`），
  沉淀流程一并收走
- 落点：本地 `~/.cache/aqa/session_trajectories/<run_id>/`（`AQA_SESSION_DIR`
  可改）；配了 `agent.knowledge.repo` 时推到结果 dataset 的 **`trajectories/`**
  （与 `knowledge/` 并列）——trajectories/runs 级别的完整会话轨迹

## 3. 同步与共享

```
本地 ~/.cache/aqa/knowledge  <-- KnowledgeSync.pull (run 开始, sha256 合并) --  HF dataset knowledge/
                          -- KnowledgeSync.push (CLOSE/failure, 只传变化文件) -->
```

- `KnowledgeSync` 把本地当权威副本，远端失败只把文件留在 `.sync_state.json` 的 pending 里，下次 run 或显式 sync 重试；不会阻断 run。
- `knowledge_sync.json` 每个 run 落盘：`status=local_only|pulled|pushed|pending|up_to_date|failed`、pulled/pushed/session 数、pending 文件与 errors；`run_manifest.knowledge` 同步记录。
- 不配 repo 就是纯本机共享（同机所有 run 共用 `default_local_dir()`，
  `AQA_KNOWLEDGE_DIR` 可改）；`agent.knowledge.sync: off|pull|push|auto|background` 控制方向；`background` 在 run 内只拉不推，push 留给 `aqa knowledge sync` 或下一次 auto run，避免 CLOSE 被网络拖住。
- Recipe KB 是**另一条 git 平面**：`recipes/quant/auto` 的 auto 配方由 `RecipeKB.pin` 写；`agent.recipe_kb.auto_sync: off|branch|pr` 可选地把它 commit 到 detached worktree 的 `aqa/kb-<run_id>` 分支（pr 再 push + `gh pr create`），默认 off。
- 上传前统一脱敏；`HfDatasetSink._stage_knowledge` 只收 `.jsonl/.patch/.json/.md`，跳过 `.sync_state.json`。

> 本 checkout 已注册 `aqa knowledge status <run_dir>` 和
> `aqa knowledge sync <run_dir> [--pull-only|--push-only]`；`aqa traces`、
> `aqa knowledge show/diff/push-references` 仍未接线。
> `KnowledgeStore.push_references()`、`session_sediment.sediment_sessions()` 是已实现的 API/内部函数。
