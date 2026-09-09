---
name: fix-and-patch
description: >
  Root-cause a failing quantize/evaluate step, fix it (env or editable source),
  retry with tier escalation, and capture a patch. Use when a pipeline step fails.
---

# Fix-and-patch skill (differentiator)

Self-heal environment/code failures during quantize or evaluate, and capture any
real source fix as a reviewable patch.

## When to use
- A quantize or evaluate step exits non-zero and the failure looks like an
  environment/toolchain/model-code/kernel bug (not a user config mistake).

## Upstream lookup playbook (不限于 auto-round)
Before theorizing a fix, check whether upstream already solved it — for ANY
relevant repo (auto-round, transformers, vllm, llm-compressor, vllm-omni,
ModelOpt, PyTorch). Deterministic queries, cheapest first:
- **最新版本**: `git ls-remote --tags https://github.com/<org>/<repo> 'v*' | tail -5`
  或 `pip index versions <pkg>`；当前安装版本 `pip show <pkg>`。
- **按错误串搜 issue/PR**: `curl -s "https://api.github.com/search/issues?q=repo:<org>/<repo>+<关键词>" | jq '.items[:5][] | {number,title,state,html_url}'`；
  或 web search `"<错误原文>" <repo> issue`。
- **定点读源码**: `curl -sL https://raw.githubusercontent.com/<org>/<repo>/<ref>/<path>`
  —— 怀疑某文件行为变更时直接读上游最新版对照本地安装版。
- **本地仓**: `/home/kaokaolv/auto_agent/auto-round` 在时优先 `git log --oneline -20`
  与 `git log <old>..HEAD -- <相关文件>`。
规则：auto-round 行为类论断必须落到其源码证据；模型结构语义类落到
transformers modeling 源码；找到现成修复（merged PR/release）优先引用并
对齐版本，不要重造轮子。所有查证结论写进 `root_cause.md` 的 EVIDENCE。

## Structured protocol (every attempt)
Emit: `COMPONENT / ERROR_CLASS / ROOT_CAUSE / EVIDENCE / VERDICT / FIX_TIER /
FIX_PLAN`. Write `root_cause.md` + `root_cause/<phase>.jsonl` every attempt.

## Tier ladder
`tiers: [openclaw, minimax, opus]`. Escalate on `VERDICT:UNFIXABLE` /
`VERDICT:ESCALATE` / drift (same error class twice). Reset drift on progress and
stay in the current tier. Respect `agent.max_fix_attempts`.

## Patch capture
- Snapshot editable deps (auto_round / vllm / model_code) **before** the agent
  edits, lazily if installed late in setup.
- Per attempt, snapshot a standalone patch (`patches/attempts/*.patch`, never
  overwritten); the final `<area>.patch` is the cumulative net diff.
- Self-check each patch by reverse dry-run apply against the pristine snapshot
  (`applies_clean`); record `resolved` (did it make the step pass) in
  `manifest.json`. Optional read-only patch review (`review.md`).

## Outputs (uploaded with results)
`root_cause.md`, `root_cause/*.jsonl`, `patches/<area>.patch`,
`patches/attempts/*`, `patches/manifest.json`, optional `review.md`.
