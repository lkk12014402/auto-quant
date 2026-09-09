# 内部 agent 的输入/输出：optimize agent 与 fix-loop

真源：`aqa/agent/optimize.py`、`aqa/agent/fix_loop.py`、`skills/fix-and-patch/SKILL.md`。
**owner 审查这些产物的路径见 §6/AGENTS.md（`logs/agent_optimize/`、
`logs/agent_fixes/<phase>/`）——把关看行为而不只是结果。**

## 1. optimize agent（`agent.optimize: true` 时，量化前）

### 输入（prompt 构成）

| 成分 | 来源 |
|---|---|
| 结构事实 | `analyze_model` 的免下载分析：architectures / model_type / is_moe / num_experts / num_layers / has_router(router_names) / has_shared_experts / has_mla / has_vision / is_diffusion / sample_modules（本地路径与 hub id 都支持） |
| 任务参数 | yaml 的 `quant.scheme / method / export_format / output_dir` |
| 操作者硬约束 | `build_quant_constraints`——**用户显式写过的字段**（scheme/method/iters/ignore_layers/layer_config/static_kv_fp8 等）逐条转成 MUST/MUST-NOT，推荐命令不得违反 |
| 知识库 | `load_recipe_knowledge`：**references 增量层优先**（`agent.knowledge.use_references` 可控），打包层兜底；diffusion 模型追加 diffusion 知识 |
| 用户上下文 | `agent.context`（内联或 `@文件`）+ skill recipe |

### 输出

| 产物 | 位置/内容 |
|---|---|
| 验证后的 auto-round 命令 | 经 `validate_command`（约束 + 专家安全规则）通过才返回；**任何一步不满足 → 返回 None，pipeline 回退确定性 builder**（optimize 只可能帮忙，永远不会阻塞） |
| `quantization_recipe.md` | 结构化分析文档（即使命令被拒也落盘，供审计） |
| `optimize.log` | 完整 agent 会话日志（`logs/agent_optimize/`） |

## 2. fix-loop（`agent.fix_loop: true` 默认开；quantize/evaluate/baseline 相位失败时）

### 触发与流程

相位失败（子进程非零 / 评估全零 / hook 判失败）→ 进入修复循环：
tier 阶梯 `tiers: [openclaw, minimax, opus]`，`max_fix_attempts: 5` 总上限、
`tier_attempt_budget: 3` 单 tier 预算、`drift_threshold: 2` 同类错误连击升级。
**每个 tier 是独立后端会话**（session 不跨 tier 继承），因此跨 tier 的关键信息
由 aqa 以结构化字段传递（见下）。

### 输入（每次尝试）

| 成分 | 说明 |
|---|---|
| 错误现场 | 失败日志尾部（含真实 traceback，由 harness 折叠尾行供签名提取） |
| 相位上下文 | model_id / scheme / method / phase 名 |
| 历史假设 | 前几轮的 `class: root_cause` 压缩行（跨 tier 的信息桥） |
| 知识库 | `KnowledgeStore`：`knowledge.replay: true` 时**先回放**已录补丁（签名匹配 error）再调 agent；lessons 注入 |
| patch 捕获 | `patch.capture: true` 时对可编辑依赖（auto_round/vllm/model_code）做前置快照 |

### 每次尝试的输出（结构化协议）

agent 必须产出 `COMPONENT / ERROR_CLASS / ROOT_CAUSE / EVIDENCE / VERDICT /
FIX_TIER / FIX_PLAN`；落盘：

| 产物 | 位置 |
|---|---|
| 尝试日志 + 会话事件 | `logs/agent_fixes/<phase>/<stem>.log`、`<stem>.events.jsonl`（截断也可审计） |
| 根因记录 | `logs/agent_fixes/<phase>/root_cause.md`、`root_cause/<phase>.jsonl` |
| 补丁 | `patches/attempts/*.patch`（单次）+ `patches/<area>.patch`（累计净 diff），每个补丁反向 dry-run 自检 `applies_clean`，`patches/manifest.json` 记 `resolved` |

### 终态

- 修复成功 → 相位重跑；`HealTally` 记录进 `run_manifest.json` 的 self_heal 段
- 修复失败 → quantize 直接抛错（run 失败，走失败回传）；evaluate 记警告并用
  最后一次结果继续（宽容策略）
- 成功修复的签名 + 补丁写回 KnowledgeStore（`knowledge.push: true` 时上传共享）

## 3. owner 审查清单

1. optimize：`logs/agent_optimize/quantization_recipe.md` 的层计划与
   `.quant_command.json` 的最终命令是否一致、是否违反你的显式配置
2. fix-loop：`logs/agent_fixes/<phase>/` 各轮的 ROOT_CAUSE/EVIDENCE 是否成立、
   补丁是否只动了该动的文件（`patches/<area>.patch` + manifest 的
   `applies_clean`/`resolved`）
3. 回放命中：knowledge 统计（run 结束日志里的 replay_hit/hit_rate）——
   高命中率说明知识库在工作；频繁 agent 修复同一错误说明知识没沉淀下来
