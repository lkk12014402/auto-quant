"""Phase → allowed-actions + legal-transition contract (RFC 0001 §S3).

This is auto-quant's Coordinator guard, modelled on Hyperloom's
``PHASE_ALLOWED_ACTIONS`` / ``allowed_actions_for`` + phase-transition machine
(``src/hyperloom/orchestrator/phases/machine_state.py``). The pipeline
(Coordinator) owns the phase sequence; a *proposer* -- the internal
fix-loop/optimize agent OR an external agent driving through ``aqa`` -- may only
act within the *current* phase's allowed-action set, and the phase chain may only
advance along the plan or take the one sanctioned loop-back (``GATE -> QUANT``,
the bounded re-QUANT; mirrors Hyperloom's ``SWEEP`` macro-cycle reloop).

Everything here is pure data + pure predicates (no IO, no agent, no GPU), so it
is fully unit-testable and safe to import anywhere.
"""

from __future__ import annotations

from ..logging_util import AqaError
from .state import Phase


class PhaseActionDenied(AqaError):
    """A proposer attempted an action/transition illegal in the current phase."""


# ── per-phase proposable actions ─────────────────────────────────────────────
# The action a proposer may take *while the Coordinator is in phase P*. ``GATE``
# holds no proposable action: the gate verdict is a Coordinator decision, never a
# proposer move (mirrors Hyperloom's COORDINATOR_RESERVED / NOT_LLM_PROPOSABLE).
PHASE_ALLOWED_ACTIONS: dict[Phase, frozenset[str]] = {
    Phase.ANALYZE:  frozenset({"target_analysis", "recipe_kb_lookup"}),
    Phase.ENV:      frozenset({"env_lock"}),
    Phase.PLAN:     frozenset({"build_plan"}),
    Phase.BASELINE: frozenset({"evaluate"}),
    Phase.QUANT:    frozenset({"quantize", "requantize", "integrate_recipe"}),
    Phase.EVAL:     frozenset({"evaluate"}),
    Phase.GATE:     frozenset(),  # pure Coordinator decision
    Phase.PUBLISH:  frozenset({"upload"}),
    Phase.CLOSE:    frozenset({"report", "pin_recipe"}),
    Phase.VERIFY:   frozenset({"verify"}),  # read-only reproduction check
}


# ── legal phase transitions ──────────────────────────────────────────────────
# Normal progress is monotonic along the chain; ``build_phase_plan`` trims which
# nodes exist per run, so forward adjacency is resolved against the plan. The ONE
# sanctioned backward edge is ``GATE -> QUANT`` -- the bounded re-QUANT loop-back
# taken on a NO-GO verdict (Hyperloom's SWEEP-exit ``cycle_reloop``).
_FORWARD_CHAIN: tuple[Phase, ...] = (
    Phase.ANALYZE, Phase.ENV, Phase.PLAN, Phase.BASELINE,
    Phase.QUANT, Phase.EVAL, Phase.GATE, Phase.CLOSE, Phase.VERIFY,
    Phase.PUBLISH,
)

_LOOPBACK_EDGES: frozenset[tuple[Phase, Phase]] = frozenset({
    (Phase.GATE, Phase.QUANT),  # bounded re-QUANT after a NO-GO verdict
})


def _coerce(phase: "Phase | str | None") -> Phase | None:
    if isinstance(phase, Phase):
        return phase
    try:
        return Phase(str(phase).strip().upper())
    except (ValueError, AttributeError):
        return None


def allowed_actions_for(phase: "Phase | str") -> tuple[str, ...]:
    """The phase's proposable actions as a sorted tuple (deterministic)."""
    key = _coerce(phase)
    if key is None:
        return ()
    return tuple(sorted(PHASE_ALLOWED_ACTIONS.get(key, frozenset())))


def is_action_allowed(action: str, phase: "Phase | str") -> bool:
    """True iff ``action`` may be proposed while the Coordinator is in ``phase``."""
    key = _coerce(phase)
    if key is None:
        return False
    return (action or "").strip() in PHASE_ALLOWED_ACTIONS.get(key, frozenset())


def is_loopback(src: "Phase | str", dst: "Phase | str") -> bool:
    """True iff ``src -> dst`` is a sanctioned (non-forward) loop-back edge."""
    s, d = _coerce(src), _coerce(dst)
    return s is not None and d is not None and (s, d) in _LOOPBACK_EDGES


def is_transition_allowed(src: "Phase | str", dst: "Phase | str",
                          plan: "list[Phase] | None" = None) -> bool:
    """True iff moving ``src -> dst`` is a legal Coordinator transition.

    Legal moves: (1) the next present phase along ``plan`` (forward progress,
    skipping trimmed-out nodes), or (2) a sanctioned loop-back edge
    (``GATE -> QUANT``). Everything else -- notably ``EVAL -> QUANT`` (a silent
    re-quant that bypasses the gate) or any other backward jump -- is denied.
    """
    s, d = _coerce(src), _coerce(dst)
    if s is None or d is None:
        return False
    if (s, d) in _LOOPBACK_EDGES:
        return (d in plan) if plan else True
    seq: list[Phase] = list(plan) if plan else list(_FORWARD_CHAIN)
    if s not in seq or d not in seq:
        return False
    return seq.index(d) == seq.index(s) + 1


def action_contract() -> dict:
    """Serialisable phase→actions + loop-back contract for the run artifact.

    Written to ``session/allowed_actions.json`` so an external proposer can read
    exactly what each phase permits (observable artifacts are the contract).
    """
    return {
        "schema_version": 1,
        "phases": {p.value: list(allowed_actions_for(p)) for p in _FORWARD_CHAIN},
        "loopback_edges": [[s.value, d.value] for (s, d) in sorted(
            _LOOPBACK_EDGES, key=lambda e: (e[0].value, e[1].value))],
        "note": (
            "Coordinator (pipeline+session) owns the phase chain and enforces these "
            "sets. A proposer (internal agent or external driver) may only act within "
            "the current phase's actions and advance along the plan, or take a "
            "sanctioned loop-back (GATE->QUANT, bounded re-QUANT)."),
    }
