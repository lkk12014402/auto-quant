"""Build the structured fix-attempt prompt for the agent.

Ports ``build_fix_prompt`` from ``local_dispatch/agent/agent_fix_loop.sh``.  The
prompt asks the agent to emit a strict, machine-parseable analysis block first
(COMPONENT / ERROR_CLASS / ... / VERDICT / SMOKE_TEST) -- this is what
:mod:`aqa.agent.verdict` parses to drive the escalation/drift state machine.
"""

from __future__ import annotations

_PROTOCOL = """\
You MUST print the block below FIRST, before changing any code or installing anything.
Do NOT modify code until you have printed an EVIDENCE_RESULT from a READ-ONLY command
that supports your hypothesis. No guessing.

COMPONENT: <our_code|transformers|auto_round|vllm|torch|model_code|data|environment>
ERROR_CLASS: <ONE stable snake_case token naming THIS error's category. Reuse the same
             token every time the same underlying error recurs (drives drift detection).>
ROOT_CAUSE_HYPOTHESIS: <one falsifiable sentence -- the specific cause>
EVIDENCE_CMD: <a single read-only command that verifies the hypothesis>
EVIDENCE_RESULT: <paste the command's output>
VERDICT: <FIXABLE | UNFIXABLE | ESCALATE>
UNFIXABLE_REASON: <required only if UNFIXABLE>
FIX_TIER: <config | upgrade | workaround | patch>   # always try the LOWEST that works
FIX_PLAN: <3 lines max -- what you will change and why it fixes the ROOT CAUSE>
SMOKE_TEST: <ONE fast command (NOT the full phase) that proves the fix works>

## Rules
- If VERDICT is UNFIXABLE: print the block and STOP (no wasted retries).
- If the fix needs a MORE CAPABLE model (deep multi-file reasoning, subtle mixed-precision
  layout, hard custom-model patch), set VERDICT: ESCALATE and STOP with a clear reason.
- Prefer the LOWEST FIX_TIER; patching source is a last resort.
- After applying the fix, RUN your SMOKE_TEST yourself and show its output before finishing.
- Read the traceback BOTTOM-UP to the exact file:line that is the real cause.
- Model custom code under ~/.cache/huggingface/modules/transformers_modules/ CAN be edited
  (e.g. replace `.float()` with `.to(other.dtype)` for dtype mismatches).
"""


def build_fix_prompt(
    phase_name: str,
    error_tail: str,
    *,
    lessons: str = "",
    attempt: int = 1,
    prior_class: str = "",
    user_context: str = "",
    prior_diff: str = "",
) -> str:
    """Assemble the full fix prompt for one attempt."""
    parts: list[str] = []
    parts.append(
        f"The `{phase_name}` phase failed (attempt {attempt}). Root-cause it from the "
        f"traceback below and fix the NARROWEST layer that owns the bug."
    )

    if user_context:
        parts.append(
            "## Operator-provided context (HIGH PRIORITY -- a human injected this; follow it)\n"
            "Treat it as authoritative for this box; prefer it over generic assumptions:\n"
            f"{user_context}"
        )

    if prior_class:
        parts.append(
            "## Quick Classification (deterministic prior -- confirm or OVERRIDE it)\n"
            f"error_class guess: {prior_class}\n"
            "Confirm against the traceback; override in your ERROR_CLASS if wrong."
        )

    if lessons:
        parts.append(
            "## Historical Lessons (from past runs -- decide which are relevant)\n"
            f"{lessons}"
        )

    if prior_diff:
        parts.append(
            "## Changes ALREADY made by earlier fix tiers (consolidate, don't restart)\n"
            "You are the STRONGEST tier and the patch CONSOLIDATOR. This unified diff is on "
            "disk now; keep what helps, fix what is wrong or incomplete:\n"
            f"```diff\n{prior_diff}\n```"
        )

    parts.append(f"## Error output (tail)\n```\n{error_tail}\n```")
    parts.append(_PROTOCOL)
    return "\n\n".join(parts)
