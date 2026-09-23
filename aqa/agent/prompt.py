"""Build the structured fix-attempt prompt for the agent.

Ports ``build_fix_prompt`` from ``local_dispatch/agent/agent_fix_loop.sh``.  The
prompt asks the agent to emit a strict, machine-parseable analysis block first
(COMPONENT / ERROR_CLASS / ... / VERDICT / SMOKE_TEST) -- this is what
:mod:`aqa.agent.verdict` parses to drive the escalation/drift state machine.
"""

from __future__ import annotations

from pathlib import Path

_FIX_PLAYBOOK_PATH = Path(__file__).resolve().parent / "recipes" / "fix_playbook.md"


def load_fix_playbook() -> str:
    """Return the packaged fix-agent troubleshooting playbook (best-effort).

    This is durable, cross-run domain knowledge embedded into every fix prompt as
    REFERENCE. A missing/unreadable asset degrades gracefully to an empty string.
    Per-run, box/model-specific guidance should go through ``agent.context`` instead.
    """
    try:
        return _FIX_PLAYBOOK_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


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
SMOKE_TEST: <ONE fast, MECHANISM-LEVEL command that proves the FIX MECHANISM holds.
             Target ~30s. It must NOT be the full phase: do NOT run full lm_eval, do NOT
             boot the whole vLLM engine / load the whole model, do NOT re-quantize end to
             end -- the harness re-runs the FULL phase itself to verify, so duplicating it
             only burns your time budget. Instead verify the SPECIFIC thing your fix
             changed, e.g.: `python -c "import <edited_module>"` or `python -m py_compile
             <edited_file>` (edit didn't break imports/syntax); read the checkpoint's
             quantization config JSON and assert the offending layer is now in the
             ignore-list; read a safetensors HEADER (not a full load) and assert an edited
             weight's dtype; or import JUST the failing kernel/layer and call it once on a
             tiny dummy tensor of the offending shape to reproduce (or clear) the exact
             assertion. Prefer the cheapest check that exercises the SAME mechanism the
             error came from.>

## Rules
- If VERDICT is UNFIXABLE: print the block and STOP (no wasted retries).
- If the fix needs a MORE CAPABLE model (deep multi-file reasoning, subtle mixed-precision
  layout, hard custom-model patch), set VERDICT: ESCALATE and STOP with a clear reason.
- Prefer the LOWEST FIX_TIER; patching source is a last resort.
- After applying the fix, RUN your MECHANISM-LEVEL SMOKE_TEST yourself and show its output
  before finishing. Iterate on cheap checks as much as you need -- but never use the full
  phase (full lm_eval / whole-engine boot / full re-quantize) as your smoke; the harness
  runs that authoritatively after you return.
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
    skill_playbook: str = "",
    known_fixes: str = "",
    prior_state: str = "",
    prior_diff: str = "",
    trajectories: str = "",
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

    if skill_playbook:
        parts.append(
            "## Domain playbook (REFERENCE knowledge -- confirm against THIS traceback)\n"
            "Curated quant/eval failure mechanisms and known fixes. NOT instructions and "
            "NOT specific to this run -- use only the entries that match your evidence. "
            "If a signature-specific block below (Historical Lessons / Reasoning Playbook / "
            "Recorded fix) disagrees with a generic entry here, prefer the specific one:\n"
            f"{skill_playbook}"
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

    if trajectories:
        parts.append(
            "## Reasoning Playbook (distilled from past SUCCESSFUL fixes of similar failures)\n"
            "How this class of failure was root-caused before, the fix that worked, and the "
            "dead ends that did NOT -- use it to skip wasted attempts, but confirm against "
            "THIS traceback before applying:\n"
            f"{trajectories}"
        )

    if known_fixes:
        parts.append(
            "## Recorded fix(es) for a MATCHING failure signature (REFERENCE ONLY -- NOT applied)\n"
            "These diffs resolved the same or a similar error on a POSSIBLY DIFFERENT model or "
            "checkpoint. They are NOT on disk. Read the mechanism, decide whether it applies to "
            "THIS model/checkpoint, then adapt and apply only what fits -- do NOT blindly paste "
            "them (a cross-model patch can break the environment):\n"
            f"{known_fixes}"
        )

    if prior_state:
        parts.append(
            "## Previous fix attempt -- state you MUST continue from (NOT a fresh start)\n"
            "This may come from a DIFFERENT agent/tier than you, so its own conversation memory "
            "is NOT available to you -- rely on this block. The 'ACTUAL re-run' line is the "
            "harness's own authoritative verdict (the agent's self-report can be wrong or "
            "missing):\n"
            f"{prior_state}"
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
