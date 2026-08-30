# 轨迹数据化（trajectories）用法

> 内核：`aqa/agent/adapters/`（多后端 → ATIF）+ `aqa/agent/traces_export.py`
> （ATIF → 数据集）+ `aqa/agent/knowledge_extract.py`（轨迹 → 知识）+
> `aqa/agent/traces_publish.py`（HF 分层归档）。设计 + 调研：
> `docs/trajectory_data_plan.md`。

## 管线总览

```
原始会话 ──> adapters ──> ATIF ──> SFT/RL 视图 ──> HF 分层归档
（openclaw/                    └──> knowledge 提取（回写 fix-loop 知识库）
 copilot/claude/codex）
```

## 1. 原始会话 → ATIF

```python
from aqa.agent.adapters import convert, detect_backend
atif = convert("~/.claude/projects/-app/<uuid>.jsonl")        # 自动探测后端
atif = convert("/logs/agent/openclaw.session.jsonl", backend="openclaw",
               model_name="nvidia/my-model")                  # openclaw 日志不带模型名
```

支持：openclaw / copilot-cli / claude-code / codex。kimi-code 上游只支持
resume、产不出 ATIF，会给明确的"不支持"错误。产物过 `validate_atif` 校验。

aqa 自己的 ReAct run（`aqa swebench` 等）直接在 run 目录写
`trajectories.jsonl`（每任务一个 ATIF dict）。

## 2. ATIF → 训练数据集（双视图）

```bash
# SFT 视图（Nemotron-Agentic 风格：reasoning_content 分离、原生 tool_calls、
# 顶层 tools 数组、每行 uuid）
aqa traces export --path results/run --view sft --out sft.jsonl

# RL 视图（NeMo-Gym 风格：responses_create_params 任务定义 + verifier 材料，
# reward 不落盘——轨迹 outcome 挂在 verifier 里供离线 RL/拒绝采样）
aqa traces export --path results/run --view rl --out rl.jsonl

# 混源 + sharegpt 列 + parquet + 推送（token 走 HF_TOKEN 环境变量）
aqa traces export --path results/run --path ~/.cache/aqa/knowledge \
    --filter all --sharegpt --parquet out.parquet --push --repo org/ds
```

源自动探测：`trajectories.jsonl` 目录 = agent_eval run；有 `trajectories/` +
`patches/` 的目录 = fix-loop 知识库。`--filter success` 只留 solved/reward>0。

## 3. 轨迹 → 知识沉淀

```bash
aqa traces distill --path results/run [--knowledge-root ~/.cache/aqa/knowledge] [--phase trajectory]
```

规则提取「错误观测 → 后续修复进展」的条目（signature/error_class/root_cause/
fix_rationale/tried_and_rejected），回写进 fix-loop 的知识库——下次别的 run
撞同类错误时回放命中。可选挂 agent 摘要（`summarizer` 可插拔），规则版是底线。

## 4. HF 分层归档

```bash
# 先本地 stage 审查
aqa traces publish --path results/run --stage-only --stage-dir /tmp/archive

# 推到 HF dataset（token 走 HF_TOKEN 或 --token-env <VAR>）
aqa traces publish --path results/run --repo org/aqa-traces
```

目录布局（每个 session 一个目录，完整会话 + 双视图并列可寻址）：

```
trajectories/<session_id>/trajectory.json    # 完整 ATIF
trajectories/<session_id>/dataset_sft.jsonl
trajectories/<session_id>/dataset_rl.jsonl
manifest.json                                # 归档清单
```

## 注意事项（调研实证）

- **episode 膨胀**：harbor 式渐增前缀切分默认关（要的话显式开），否则 N 倍膨胀；
- `is_copied_context` 的 step 必须剔除出 SFT（ATIF 规范义务）；
- RL 视图不含回答——回答是 rollout 时在线生成的；我们挂的是 outcome 元数据。
