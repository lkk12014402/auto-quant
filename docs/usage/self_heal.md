# Agent 自愈（self-heal）用法

> 内核：`aqa/agent/`（fix_loop / verdict / drift / patch / knowledge / optimize）。
> 判断层：`skills/fix-and-patch/SKILL.md`。

## 最小配置

```yaml
agent:
  fix_loop: true          # 相位失败（量化/评估）时自动修复重试
  optimize: true          # 量化前让 agent 分析模型并产出最优命令
  tiers: [openclaw, minimax, opus]   # 升级阶梯（便宜在前）
  max_fix_attempts: 5
  tier_attempt_budget: 3  # 每层尝试预算，用尽升级
  drift_threshold: 2      # 同一 error_class 连续 N 次 = 漂移升级
  knowledge:
    enabled: true
    repo: org/aqa-knowledge    # 可选：HF dataset 跨机器共享知识
    # local_dir: ~/.cache/aqa/knowledge   # 不配 repo 时的本地库
```

## 工作方式

1. **结构化协议**：agent 必须输出 `COMPONENT / ERROR_CLASS / ROOT_CAUSE /
   EVIDENCE / VERDICT / FIX_PLAN / SMOKE_TEST`，解析严格（散文/占位符不算数）；
2. **确定性状态机**：升级/漂移/预算都是 Python 判定，不靠 agent 自觉；
3. **patch 捕获**：修复前先给可编辑依赖（auto_round/vllm/模型自定义代码）打
   原始快照，agent 的改动收拢成可审查的 unified diff（`patches/`，带
   `applies_clean` 反向校验和 `resolved` 记录）；
4. **知识回放**：每个成功修复按**错误签名**（不是按模型）存进知识库；下次
   撞同类错误，先查库——补丁能干净应用且让相位通过就**零 agent 调用修复**，
   否则回滚快照让 agent 上。配 `knowledge.repo` 可跨机器共享。

## 环境依赖

- tier 后端：openclaw（`npm i -g openclaw`，`MINIMAX_API_KEY`）/ copilot
  （`@github/copilot`，`COPILOT_GITHUB_TOKEN`）；
- 容器/venv 路径自动供给（见 `docs/usage/dispatch.md`）；本机直跑要自己装；
- 某个 tier 的 CLI 或 token 缺失时自动跳过该 tier 继续升级，不会卡死。

## 产物

run 目录下：`root_cause.md`（人读根因报告）+ `root_cause/<phase>.jsonl`
（机读记录）+ `patches/`（attempts + 最终 patch + manifest）+
`logs/agent_fixes/`（每轮尝试的 prompt/日志）+ `logs/agent_optimize/`（
optimize 轨迹：events.jsonl + share.md）。

## 与轨迹数据化的衔接

fix-loop 的成功修复会沉淀进知识库；知识库里的摘要可以转成 SFT 训练数据
（`aqa traces export --path <knowledge_root>`），也可以从 agent 轨迹直接提取
「错误→修复」模式回写知识库（`aqa traces distill`，见 `docs/usage/trajectories.md`）。
