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
