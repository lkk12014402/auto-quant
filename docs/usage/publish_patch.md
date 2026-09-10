# 补丁转 PR（`aqa publish-patch`）

把 run 中沉淀的补丁变成 fork 上的真实 PR——auto-round 场景默认：
fork `lkk12014402/auto-round`，上游 `intel/auto-round`。**真源：`aqa/cli.py`
的 `publish_patch`。**

## 前置条件

| 项 | 说明 |
|---|---|
| fork 本地仓 | `--repo-dir` 指向它（origin=fork，upstream 缺失会自动添加）；**它自己的脏工作区不会被碰**——所有操作在临时 worktree 里做 |
| `gh` CLI | PR 创建靠它；没有就直接报错，不会静默做成"只推了分支" |
| token | **只从环境读**：`gh auth` 或 `GH_TOKEN`/`GITHUB_TOKEN`；禁止写进 URL/文件 |

## 流程（代码顺序）

```
① git fetch upstream <base>            # 上游最新
② git push origin upstream/<base>:<base>   # fork 基线同步（sync fork）
③ git worktree add --detach <tmp> upstream/<base>   # 临时工作区，不动本地仓
④ git apply --check <patch>            # 打不上就停，不产生任何远端动作
⑤ git apply + add + commit             # commit message = PR 标题 + 正文
⑥ git push -u origin <branch>          # 分支默认 aqa/<补丁名>-<日期>
⑦ gh pr create --repo <fork> --base <base> --head <branch>
```

## 用法

```bash
# ① 先 dry-run（不联网）：打印计划 + 在本地已 fetch 的 upstream 树上真验 apply
aqa publish-patch patches/owner/resolver_guard.patch \
    --repo-dir /home/kaokaolv/auto_agent/auto-round \
    --title "fix: guard scheme-attr stripping in apply_plan_to_model" \
    --dry-run

# ② 正式发
aqa publish-patch patches/owner/resolver_guard.patch \
    --repo-dir /home/kaokaolv/auto_agent/auto-round \
    --title "fix: guard scheme-attr stripping in apply_plan_to_model" \
    --body "动机：Qwen4Exp 的 RMSNorm.group_size 被误剥……证据：……测试：……"
```

参数：`--upstream`（默认 intel/auto-round）、`--base`（默认 main）、
`--fork-repo`（默认从 origin URL 解析）、`--branch`（自定义分支名）。

## 与 run 产物的关系

- 补丁来源：`patches/owner/<label>.patch`（`aqa capture-patch` 捕获的 owner
  手改）或 `patches/<area>.patch`（fix-loop 自产、经你审核接受的）
- 惯例：**先发 PR 再记决策**——

```bash
aqa log-action <run_dir> --kind decision \
  --text "resolver guard 补丁已转 PR <url>" --command "aqa publish-patch ..."
```

- PR 正文建议带上证据（动机/根因/测试）——内容直接取 `logs/agent_fixes/`
  的 root_cause 与你的审核记录

## 排错

| 现象 | 原因/处理 |
|---|---|
| `patch does not apply on upstream/main` | 上游已变，补丁过期 → 用最新 upstream 重新生成补丁（capture-patch 重抓） |
| `cannot parse fork owner/name` | origin 不是 GitHub URL → 显式 `--fork-repo owner/name` |
| `gh pr create failed` | token 权限/过期——`gh auth status` 检查 |
| fork 同步被拒 | fork 的 main 有本地独有提交——先确认那些提交要不要合入上游 |

## 纪律

- PR 总是**基于同步后的上游 main**（流程里内建了，不用手工 sync fork）
- 一个 PR 一个主题补丁；混合改动拆开发
- 补丁进 PR 前必须先过 owner 审核（`docs/usage/owner_interface.md` 的审核协议）

## 实战案例（2026-09-10，真实运行）

vllm-omni MXFP8 修复：owner agent 在 B300 上定位 `OmniINCConfig.get_quant_method`
的 MXFP8 特判误伤 text encoder（`block_name_to_quantize` 未检查），patch 落
`/models/auto-quant/patches/fix_mxfp8_text_encoder_quantization.patch`；kimi 独立验证
（逐层统计 + 图像内容三对照）后执行：

```bash
GH_TOKEN=$(git -C <vllm-omni仓> remote get-url origin | sed -E 's|.*https://([^@]+)@.*|\1|') \
  aqa publish-patch <patch文件> \
    --repo-dir <vllm-omni仓> --upstream https://github.com/vllm-project/vllm-omni \
    --base main --branch fix_mxfp8_model_load \
    --title "[Bugfix][Quant] Scope AutoRound MXFP8 offline method to block_name_to_quantize" \
    --body "<问题/修复/验证>"
# 产出：https://github.com/lkk12014402/vllm-omni/pull/1（fork 内 PR，人工审后发上游）
```

途中踩坑与处理（都有教学价值）：

1. **patch 在传输/手写中损坏**（hunk 行数不符，`corrupt patch at line N`）——
   不要手修 hunk，直接在干净 checkout 上重做修改再 `git diff` 重新生成
2. **重跑撞残留分支**（上次失败留下的 `aqa/...` 分支）——`git branch -D` +
   `git worktree prune` 清掉再来
3. **机器没装 gh**——`publish-patch` 的 PR 步需要它；装到用户目录
   （`~/.local/bin/gh`）即可，不要污染系统
