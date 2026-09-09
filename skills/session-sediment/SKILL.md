---
name: session-sediment
description: >
  Owner wrap-up flow after an aqa run: sediment the owner agent's own session
  (and verify aqa's internal sessions) into the shared trajectories store.
  Use at the END of any aqa-driven quantization/eval task.
---

# Session sediment (owner 收尾流程)

run 结束时把**本次任务所有 agent 的 session** 沉淀为 ATIF 轨迹并回传——
量化知识只是经验的一半，session 里的排错过程才是可训练的资产。

## 分工（谁沉淀谁的）

| session | 谁负责 | 机制 |
|---|---|---|
| aqa fix-loop（`logs/agent_fixes/**/*.events.jsonl`） | **aqa pipeline 自动** | run 收尾自动转换（`agent.sediment_sessions: true` 默认开） |
| aqa optimize（`logs/agent_optimize/*.events.jsonl`） | **aqa pipeline 自动** | 同上 |
| **owner 自己**（copilot/kimi/... 的本次会话） | **owner（本 skill）** | aqa 碰不到 owner 的会话文件，只能你来 |

## Owner 收尾步骤

1. **找到你自己的 session 文件**（按你使用的 CLI）：
   - copilot：`~/.copilot/session-state/<id>/events.jsonl`
   - kimi：`~/.kimi-code/sessions/<wd>/<session>/agents/main/wire.jsonl`
   - 其它：adapters 支持 claude / codex / openclaw（`aqa/agent/adapters/`）
2. **放进 run 目录的 `sessions/`**（命名带 backend 提示，如 `copilot_xxx.jsonl`
   / `kimi_wire.jsonl`——自动识别靠文件名/内容嗅探）：

```bash
mkdir -p <run_dir>/sessions
cp <我的session文件> <run_dir>/sessions/
```

3. **触发沉淀**（也可以等下一次 run 自动带上，但显式触发更干净）：

```bash
python -m aqa.agent.session_sediment  # 或重跑任意收尾命令让 pipeline 带上
```

也可直接离线转换验证：`aqa traces convert --path <session文件> --out /tmp/check`
看步数是否正常（<2 步说明文件不对）。

4. **确认回传**：配了 `agent.knowledge.repo` 时会推到结果 dataset 的
   `trajectories/`（与 `knowledge/` 并列）。本地暂存在
   `~/.cache/aqa/session_trajectories/<run_id>/`（`AQA_SESSION_DIR` 可改）。

## 纪律

- **脱敏是自动的**（适配器收口处执行，规则集 `aqa/report/scrub.py`），
  但放进 sessions/ 前自己扫一眼有没有明文密钥残留更稳妥
- session 里如果有**不想上传的内部路径/讨论**，不要放 sessions/——
  沉淀是默认全量上传的
- 收尾顺手把补丁处置了：值得上游的 `aqa publish-patch`（先过审核协议）
