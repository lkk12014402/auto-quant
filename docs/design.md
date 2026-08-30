# aqa 设计文档（autoquant-agent）

> 一页看懂 aqa 的架构、数据流和设计决策。模块级用法见 `docs/usage/*.md`；
> 各专题深挖见 `docs/` 下的 deep-dive / plan / recipe 文档。

## 1. 它是什么

aqa 把「一个 `run.yaml`」变成一条可复现的 `量化 → 评估 → 门禁 → 归档` 流水线，
并且自带两层增强：**agent 自愈**（fix-loop 定位并修复失败，产出可审查补丁）与
**轨迹数据化**（agent 轨迹 → ATIF → SFT/RL 训练数据集 → 知识沉淀 → HF 归档）。

## 2. 架构总览

```
                         ┌─────────────────────────────┐
                         │          run.yaml            │  单一声明式入口
                         └──────────────┬───────────────┘
                                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ pipeline（aqa/pipeline.py）                                           │
│                                                                       │
│  provider ──> quantize ──> evaluate ──> gate ──> sinks                │
│  （pretrained│quantize│endpoint）  ▲        │                         │
│                                   │        │ accuracy.json            │
│            ┌── agent fix-loop ◀───┘        │ run_manifest.json         │
│            │  (tier 升降级 / 知识回放 /      ▼ gate.json               │
│            │   patch 捕获)                local_fs / hf_dataset /      │
│            │                               hf model repo              │
│            ▼                                                          │
│  trajectory 数据化管线（aqa/agent/）                                   │
│  原始会话 ──> adapters ──> ATIF ──> SFT/RL 视图 ──> HF 分层归档        │
│     （openclaw/copilot/claude/codex）        └──> knowledge 提取       │
└───────────────────────────────────────────────────────────────────────┘
```

三层职责分离：

- **内核（`aqa/`）**：纯 Python 确定性状态机——配置、注册表、流水线、执行器、
  评估、sink。全部可单测，唯一出壳的是容器内命令执行。
- **判断层（`skills/`）**：agent 的领域知识（量化 scheme 选择、基准配置、
  修复策略），以 SKILL.md 形式注入 agent，不进内核。
- **数据层（trajectory 管线）**：把 agent 的劳动沉淀成训练资产（数据集 +
  知识库），反哺判断层。

## 3. 关键设计决策（为什么长这样）

1. **评估为核心，量化是 provider 插件**（`provider.type: quantize`）。
   评估一个现成模型、一个量化产物、一个在线 endpoint，同一条流水线。
   量化不是框架存在的理由，评估才是——这让 aqa 同时是量化工具和评测框架。

2. **一切经 entry-point 注册表**（providers/harnesses/backends/executors/
   sinks/scorers/hooks/recipe_stores）。加能力 = 发一个小 pip 包声明
   entry-point，核心仓不需要 fork。

3. **配置单解析点**（`config.py`）：`_base_` 继承 + 深度合并 + `${VAR}` 环境
   展开；secrets 只许 `*_env` 引用，永不内联——开源硬约束。

4. **per-benchmark 参数钉在 recipe**（`recipes/benchmarks/<task>.yaml`），
   run.yaml 只管全局策略。任务的正确配置跟着任务走，不跟着每次 run 走。

5. **执行器三载体同构**（local venv / local docker / ssh docker）：共享同一个
   容器 bootstrap（装栈 + agent 供给 + 内层 `aqa run`），secrets 同源
   （`collect_secret_env` 白名单 + sink token 引用），sinks 在内层 run 里
   统一发生。切载体只改 yaml 的 `executor` 块。

6. **agent 自愈是确定性状态机**（drift/budget/escalation 都是 Python），
   agent CLI 只在 tier 里被调；补丁先快照后捕获、可反向校验
   （`applies_clean`）；知识库按错误签名回放，命中就零 agent 调用修复。

7. **轨迹即资产**：ATIF（Harbor 规范）作中间格式；SFT 视图对齐
   Nemotron-Agentic-v1（reasoning_content 分离、原生 tool_calls）；
   RL 视图对齐 NeMo-Gym（任务定义 + verifier 材料，reward 不落盘——
   rollout 时环境给）；知识沉淀把「错误→修复」模式回写进 fix-loop 的
   回放库。详见 `docs/trajectory_data_plan.md`。

## 4. 模块地图

| 模块 | 职责 | usage 文档 |
|---|---|---|
| `config.py` | run.yaml 解析/组合/校验 | `docs/usage/quantize.md` |
| `providers/` | 被评模型从哪来 | 同上 |
| `quantize/` | auto-round 命令构建与执行 | 同上 |
| `eval/` | harness（lm_eval/native/diffusion）+ 打分器 + runner | `docs/usage/evaluate.md` |
| `backends/` | hf / vllm / vllm_omni 推理后端 | 同上 |
| `executors/` | local(current/venv/docker) / ssh | `docs/usage/dispatch.md` |
| `report/` | sinks（local_fs/hf_dataset）+ gate | `docs/usage/evaluate.md` |
| `agent/` | fix-loop + patch + knowledge + optimize + 轨迹数据化 | `docs/usage/self_heal.md` + `docs/usage/trajectories.md` |
| `agent_eval/` | 端点评估 / ReAct / sandbox / ATIF 导出 | `docs/usage/trajectories.md` |
| `hooks/` | 相位钩子（quant_sanity / eval_sanity） | `docs/usage/evaluate.md` |
| `skills/` | agent 判断层（quantize/evaluate/fix-and-patch） | — |
| `recipes/` | 钉死的 benchmark/scheme/quant recipe | — |
| `docs/` | 专题深挖（day-0 / diffusion / nemo / codex / deepseek / 轨迹） | — |

## 5. 扩展一个能力的标准动作

- 新 benchmark：丢一个 `recipes/benchmarks/<name>.yaml`（需要代码打分时再加
  一个 scorer 插件）；
- 新推理后端/执行器/sink/harness/scorer/hook：写一个声明 entry-point 的小包；
- 新量化 scheme：丢一个 `recipes/schemes/<name>.yaml`；
- 新 agent 后端轨迹源：`aqa/agent/adapters/<backend>.py` 一个 `to_atif` +
  注册表一行；
- 新模型架构（day-0）：走 `docs/day0_quantization_recipe.md` 的 playbook。

## 6. 专题深挖文档索引

| 文档 | 内容 |
|---|---|
| `docs/day0_quantization_recipe.md` | 新架构 day-0 量化 playbook + Qwen3.8-Flash-Next 实证 |
| `docs/diffusion_support_plan.md` | diffusion 量化支持的设计与里程碑 |
| `docs/autoround_diffusion_reference.md` | auto-round diffusion 机制 + vllm-omni 部署矩阵（源码级） |
| `docs/autoround_full_reference.md` | auto-round scheme/算法/CLI 全表（源码级） |
| `docs/trajectory_data_plan.md` | 轨迹数据化的调研（harbor/NeMo）与设计 |
| `docs/nemo_adoption_design.md` | 借鉴 NeMo Evaluator/Skills 的采用设计 |
| `docs/nemo_skills_deep_dive.md` | NeMo Skills inference/evaluation/dataset 深挖 |
| `docs/codex_harness_deep_dive.md` | Codex harness 深挖 |
| `docs/deepseek_eval_deep_dive.md` | DeepSeek 评估深挖 |
