# Trajectory Export Demo — 多 backend 真实会话的轨迹数据化产物

本目录是 `aqa traces` 轨迹数据化链路的端到端实测产物，按 `<backend>_<session-id 前 8 位> 分子目录：

## 目录布局

```
demo/trajectory_export/
├── kimi_7f851d8a/            # Kimi Code CLI 主会话（本机长任务会话，4604 步、
│   │                         # 876 轮用户交互，含 aqa 全部 feature 建设、
│   │                         # INC-Quantization-Recipes 回刷方案与 12 份 owner
│   │                         # 任务书、远程 k8s llm-server 的 owner 修复 review
│   │                         # 与 merge；agent-0..27 子代理轨迹 ×28）
│   ├── trajectories.jsonl    # ATIF 轨迹
│   ├── sft.jsonl / rl.jsonl / sharegpt.jsonl
│   ├── archive/              # 分层发布包（trajectories/<sid>/ 三件套 + manifest）
│   └── knowledge/trajectory.jsonl
├── kimi_974fed50/            # 以下来自 /home/kaokaolv/kimi_pt（wd_hshen 工作区，
├── kimi_418a9551/            # 42 条 wire.jsonl 全部转换成功）
├── kimi_fe4af318/            # 主会话（2096 步；含大型多代理任务）
├── kimi_fe4af318_agent-*/    # 同会话的子代理轨迹 ×23（适配器给子代理的
│                             # session_id 加代理名后缀消歧，否则归档目录撞名）
├── kimi_8df5de67*/
├── kimi_800407d2/            # kimi 会话（wd_my_commit 工作区，253 步、53 轮，
│                             # copilot 自定义 API 配置任务；脱敏实证：会话里贴的
│                             # Access Token 已被规则正确掩码）
├── copilot_<id8>/            # GitHub Copilot CLI 会话 ×10（来自容器 lkk-ki 的
│                             # ~/.copilot/session-state/*/events.jsonl）
├── copilot_a397b109/         # muse_support 迁移工作会话完整版（B300 整机
│                             # ~/copilot_B300，2576 步、106 轮用户交互；
│                             # 取代 09-03 从 ~/Pai-Megatron-Patch 导出的
│                             # 668 步部分版）。内容：muse 插件 0.17→main
│                             # 迁移 + EP all-reduce 根因 + MXFP4/W4A16
│                             # 在新栈上的使能 + 以 nvfp4 为锚的 logprob
│                             # 数值验证方法论 + 打包分支与 XPU 分支合并
│   ├── trajectories.jsonl
│   ├── sft.jsonl / rl.jsonl / sharegpt.jsonl
│   └── knowledge/trajectory.jsonl   # 仅当该会话有 error→fix 条目时存在
└── README.md
```

容器 `lkk-ki` 共 32 个 session 目录，其中 12 个有 `events.jsonl`，2 个是
「启动即关闭」的空会话（无对话内容，适配器正确拒绝），实际沉淀 10 个。
全部产物已过统一脱敏（`aqa/report/scrub.py` 规则集，适配器收口处自动执行）。

## 复现命令

```bash
# 1. 会话日志 → ATIF（backend 自动识别；文件名不含 backend 名时
#    用 --backend 强制指定，如 copilot 的 events.jsonl）
aqa traces convert --path <session.jsonl> [--backend copilot] --out <out_dir>

# 2. ATIF → 数据集视图
aqa traces export --path <out_dir> --view sft --out sft.jsonl
aqa traces export --path <out_dir> --view rl  --out rl.jsonl
aqa traces export --path <out_dir> --view sft --sharegpt --out sharegpt.jsonl

# 3. 知识蒸馏（--knowledge-root 省略时写共享库 ~/.cache/aqa/knowledge/）
aqa traces distill --path <out_dir> --knowledge-root <out_dir>/knowledge

# 4. 分层发布包（去 --stage-only 加 --repo org/name 即推 HF dataset）
aqa traces publish --path <dir1> --path <dir2> ... --stage-only --stage-dir archive
```

## 说明

- 会话不含 solved/reward 元数据，走"无 outcome 保留"路径；fix-loop 产出的
  轨迹才有 success 过滤的实际效果。
- 知识条目为规则抽取的保底下限；接 agent summarizer 可进一步提炼
  root_cause / fix_rationale。
- copilot 会话的 session_id / agent_version 取自 `session.start` 事件
  （适配器已增强；旧版无此事件的日志回退 Harbor 惯例 "copilot-cli"）。
