# 轨迹数据化产品化设计：多后端 agent 轨迹 → 训练数据集 + 知识沉淀 + HF 分层上传

> 2026-08-29。目标：把 aqa 现有的轨迹转化能力（`aqa/agent_eval/atif.py` +
> `aqa/agent/traces_export.py`）产品化为一条完整管线：多后端 agent 原始会话
> → 统一 ATIF → 双视图训练数据集（SFT/RL）→ 知识沉淀 → HF dataset 分层归档。
> 调研依据：harbor-framework/harbor（ATIF 规范与后端适配架构）与
> NVIDIA NeMo-RL / NeMo-Gym / Nemotron（训练数据格式）。

## 1. 现状盘点（aqa 已有）

| 模块 | 现状 |
|---|---|
| `agent_eval/atif.py` | aqa 自有 ReAct 轨迹 → ATIF-v1.7 dict → `trajectories.jsonl`；轻校验器 |
| `agent/traces_export.py` | ATIF jsonl → OpenAI messages 行（SFT）；fix-loop 知识摘要 → SFT 行；sharegpt 列；jsonl/parquet；`push_to_hub`（平铺 dataset） |
| fix-loop knowledge | 成功修复的结构化摘要 + patch diff（`trajectories/*.jsonl` + `patches/`） |

**缺口**：
1. 多后端原始会话 → ATIF 的适配器层缺失（只有 aqa 自有 ReAct；openclaw/copilot 的会话日志没有进 ATIF；claude/codex/kimi 没有）；
2. 数据集只有 SFT 单视图（messages），没有 RL 视图（NeMo-Gym 式任务+校验材料），也没有 reasoning_content 分离、tool 原生结构（现在 observation 被摊平成 tool 角色文本）；
3. 上传是平铺 dataset push，没有 `trajectories/<session_id>/` 的目录化归档（完整会话 + 轨迹 + 数据集 + 知识分层）；
4. 知识沉淀只有 fix-loop 摘要，没有「轨迹 → agent 摘要 → 知识条目」的通用提取管线。

## 2. 调研结论（设计依据）

### 2.1 Harbor（ATIF 是规范，适配是分散式生产者）

- ATIF 规范：`rfcs/0001-trajectory-format.md`（v1.8 当前），Pydantic 参考实现
  `src/harbor/models/trajectories/`。关键字段：Trajectory{schema_version,
  session_id, agent, steps[], final_metrics, extra}；Step{step_id, source
  (system/user/agent), message, reasoning_content, tool_calls[], observation,
  metrics{prompt_tokens/completion_tokens/cost_usd, **RL 预留 logprobs +
  token_ids**}, is_copied_context, llm_call_count}；ToolCall{tool_call_id,
  function_name, arguments}；ObservationResult{source_call_id, content}。
- 后端适配架构：**没有统一转换层**——每个 backend 一个 `_convert_*_to_atif`
  （claude-code 解析 ~/.claude session jsonl、codex 解析 rollout-*.jsonl、
  gemini-cli 读 trajectory.jsonl、copilot-cli 读自己的 jsonl……），共享的只有
  Pydantic 模型 + 能力位声明（`AgentCapabilities(atif=True)`）。支持 ATIF 的
  后端 30 个（含 claude-code/codex/copilot-cli/openclaw；kimi-code 只支持 resume）。
- `harbor traces export`：ATIF → OpenAI conversations 的 episode 渐增前缀切分
  （每个 agent step 一条训练样本）；`is_copied_context` step 必须从 SFT 剔除；
  observation 映射为 user 角色、tool call 内联成 `<tool_call>` 文本标签
  （Qwen 风格）。**坑：不做去重会 N 倍数据膨胀**。
- 上传：HF 通道是平铺 dataset push；harbor 自有云服务才有
  `trials/<trial_id>/trajectory.json` 的目录化组织（轨迹单独直传 + 整包归档）。

### 2.2 NeMo（RL 数据不放 reward，SFT 用 OpenAI 原生 tool-calling）

- **SFT**（Nemotron-Agentic-v1 实物）：JSONL `{uuid, messages, tools}`，
  messages 是 OpenAI 原生 tool-calling 结构（assistant 的 `tool_calls` 数组 +
  `role:"tool"` 结果消息），`reasoning_content` 单独字段（thinking 不进
  content）。loss mask：assistant=1，其余=0。
- **RL**（NeMo-Gym 协议，Nemotron-3 实际在用）：JSONL 每行一个**任务实例**，
  `responses_create_params`（初始 input + tools + sampling 参数——注意：
  **prompt 只有初始输入，不含回答**，回答是 rollout 时在线生成的）+
  verifier 材料（`ground_truth`/`expected_answer`/`verifier_metadata`）+
  `environment_name` + `agent_ref`。异构 schema 按 `dataset`/`environment_name`
  字段区分，不强行统一列。
- **reward 不落盘**：环境按 turn 返回 reward，rollout 累加成 trajectory 级
  `total_reward`，advantage 训练时算。数据里最多放 `pass_rate` 难度元数据。
- 已有轨迹（离线）的 RL 化：数据集行 = 任务定义 + verifier 材料 + 我们把
  轨迹的 outcome（solved/reward）挂进 verifier 材料区，供离线 RL/拒绝采样用。

### 2.3 格式选型（双视图）

- **SFT 视图**：Nemotron-Agentic-v1 风格 `{uuid, messages, tools,
  reasoning_content 分离}`——OpenAI 原生 tool_calls + tool 角色，不再摊平文本。
- **RL 视图**：NeMo-Gym 风格 `{responses_create_params, verifier_materials,
  environment_name, agent_ref, extra: {solved, reward}}`。
- 中间格式：ATIF 不变（已是我们的交换格式，harbor 生态兼容）。

## 3. 目标架构

```
原始会话（多后端）                    ATIF（统一中间格式）              产出
─────────────────                  ──────────────────              ──────
aqa ReAct trajectories  ──┐
openclaw session jsonl  ──┤
copilot session jsonl   ──┼──> adapters/<backend>.py ──> ATIF dict ──┬─> SFT 视图（messages+tools+reasoning_content）
claude session jsonl    ──┤    （逐 backend 一个适配器，   校验器     ├─> RL 视图（responses_create_params + verifier）
codex rollout jsonl     ──┤     共享 ATIF 模型）                    ├─> sharegpt 列（兼容）
kimi session（如可得）  ──┘                                         └─> knowledge 提取（agent 摘要 → 知识条目）
                                                                            │
                                        HF dataset 分层归档（push_to_hub 目录模式）：
                                        trajectories/<session_id>/trajectory.json（完整 ATIF）
                                        trajectories/<session_id>/dataset_sft.jsonl
                                        trajectories/<session_id>/dataset_rl.jsonl
                                        knowledge/<signature>/...（知识条目）
```

**适配器层**（`aqa/agent/adapters/`）：每后端一个 `to_atif(raw) -> dict`，
注册表 `ADAPTERS`，能力声明 + 自动探测（按文件特征识别后端）。写入即校验
（复用 `atif.validate_atif`）。无本地样本的后端用 harbor 转换器的格式契约
做合成 fixture 测试。

**数据集转化**（增强 `traces_export.py`）：
- SFT 视图重写：reasoning_content 分离、tool_calls 原生结构、observation →
  `role:"tool"`（不再是 user/文本内联）——对齐 Nemotron-Agentic-v1；
- RL 视图新增：任务 input + tools + verifier_materials（含轨迹 outcome）；
- 保留现有 sharegpt / fixloop 行兼容。

**知识沉淀**（新 `aqa/agent/knowledge_extract.py`）：从 ATIF 轨迹提取候选
（错误→修复模式、有效命令序列、死胡同），可选走 agent 摘要，写回
knowledge store（trajectories/*.jsonl + patches/），与 fix-loop 知识同构。

**HF 分层上传**（`traces_export.push_to_hub` 增强为目录模式）：
`trajectories/<session_id>/` 下放完整 ATIF + 双视图 jsonl；knowledge 单独
子树；token 只从环境取（HF_TOKEN / token_env）。

## 4. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 适配器层 | `aqa/agent/adapters/`：openclaw / copilot / claude / codex 的 session→ATIF 适配器 + 自动探测 + 合成 fixture 测试 | 每个适配器对合成样本产出通过 `validate_atif` 的 ATIF |
| M2 双视图数据集 | SFT 视图重写（reasoning_content + 原生 tool_calls）+ RL 视图（NeMo-Gym 式）+ sharegpt 保留 | 合成 ATIF → 两视图 jsonl，字段级单测 |
| M3 知识沉淀 | 轨迹 → 知识条目提取（规则提取先行，agent 摘要可选）；写回 knowledge store | 合成轨迹 → 知识条目单测 |
| M4 分层上传 | HF dataset 目录模式上传（trajectories/<session_id>/...） | 本地 staging 目录结构单测（不真传） |

## 5. 风险与边界

- **无本地真实样本**：本机没有 openclaw/copilot/claude/codex 的会话日志（容器/远端跑的）——适配器按 harbor 转换器的格式契约开发 + 合成 fixture；首次遇到真样本要回归校验。
- kimi-code 在 harbor 里只有 resume 支持（无 ATIF 产出）——标记为"暂不支持，留适配器接口"。
- 数据膨胀：episode 渐增前缀切分默认关（`--episodes last` 只导最终完整对话），需要时显式开。
