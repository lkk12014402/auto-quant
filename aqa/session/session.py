"""The ``Session`` facade: binds the state machine, manifest and event stream.

``open_session`` is the single entry the pipeline uses to create a fresh session
(writing ``manifest.json`` + an initial ``state.json``) or to resume one
(refusing when the manifest is missing, reloading the prior ``state.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .actions import (
    PhaseActionDenied,
    action_contract,
    allowed_actions_for,
    is_action_allowed,
    is_loopback,
    is_transition_allowed,
)
from ._io import atomic_write_json
from .events import append_event
from .manifest import build_session_manifest, load_session_manifest, write_session_manifest
from .state import Phase, SessionState

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

_SESSION_DIR = "session"


def _as_phase(phase: "Phase | str | None") -> Phase | None:
    if isinstance(phase, Phase):
        return phase
    try:
        return Phase(str(phase).strip().upper())
    except (ValueError, AttributeError):
        return None


@dataclass
class Session:
    """Bundle a run's ``session/`` dir with its live :class:`SessionState`.

    Every transition commits ``state.json`` atomically *and* appends an event, so
    the on-disk record is always consistent with where the pipeline actually is.

    The Coordinator guard (RFC §S3) rides here: ``guard_action`` checks a proposer
    action against the *current* phase's allowed-action set, and ``enter`` annotates
    each phase transition as forward / loop-back / irregular.
    """

    dir: Path
    state: SessionState
    guard_enabled: bool = True   # record phase_action_denied events
    guard_enforce: bool = False  # also raise PhaseActionDenied on a denied action

    def commit(self) -> None:
        self.state.commit(self.dir)

    def event(self, type: str, **fields) -> None:
        append_event(self.dir, type, **fields)

    def enter(self, phase: Phase) -> None:
        prev = _as_phase(self.state.phase)
        plan = [p for p in (_as_phase(n) for n in self.state.phases) if p is not None]
        allowed = prev is None or prev == phase or is_transition_allowed(prev, phase, plan)
        self.state.enter(phase)
        self.commit()
        self.event("phase_enter", phase=phase.value)
        if prev is not None and prev != phase:
            self.event(
                "phase_transition",
                **{"from": prev.value, "to": phase.value,
                   "allowed": allowed, "loopback": is_loopback(prev, phase)},
            )

    def done(self, phase: Phase, **info) -> None:
        self.state.done(phase)
        self.commit()
        self.event("phase_done", phase=phase.value, **info)

    def skip(self, phase: Phase, reason: str = "") -> None:
        self.state.skip(phase)
        self.commit()
        self.event("phase_skip", phase=phase.value, reason=reason)

    def fail(self, phase: Phase, error: str = "") -> None:
        self.state.fail(phase)
        self.commit()
        self.event("phase_fail", phase=phase.value, error=error)

    def guard_action(self, action: str, phase: "Phase | str | None" = None,
                     **ctx) -> bool:
        """Check a proposer action against a phase's allowed-action set (RFC §S3).

        ``phase`` defaults to the session's *current* phase, so any caller
        (internal fix-loop/optimize agent, the requant loop-back, or an external
        agent routed through ``Session``) is held to what that phase permits.
        Returns ``True`` when allowed. When disallowed and the guard is enabled,
        records a ``phase_action_denied`` event (+ a decision); when enforcing,
        raises :class:`PhaseActionDenied`. Mirrors Hyperloom's per-phase
        ``allowed_actions`` whitelist enforced at dispatch.
        """
        cur = _as_phase(phase) if phase is not None else _as_phase(self.state.phase)
        if not self.guard_enabled or is_action_allowed(action, cur):
            return is_action_allowed(action, cur)
        allowed = allowed_actions_for(cur)
        cur_name = cur.value if isinstance(cur, Phase) else str(cur)
        self.event("phase_action_denied", phase=cur_name, action=action,
                   allowed=list(allowed), enforced=self.guard_enforce, **ctx)
        if isinstance(cur, Phase):
            self.state.record_decision(
                cur, what=f"deny action '{action}'",
                why=f"not allowed in phase {cur_name}; allowed={list(allowed)}")
            self.commit()
        if self.guard_enforce:
            raise PhaseActionDenied(
                f"action '{action}' is not allowed in phase {cur_name} "
                f"(allowed: {', '.join(allowed) or 'none'})")
        return False

    def should_skip(self, phase: Phase, artifact_present: bool) -> bool:
        """Resume rule (RFC §8): a phase is skippable only when a prior run marked
        it ``DONE`` *and* its product is still on disk; a half-baked artifact
        re-runs the phase."""
        return self.state.is_done(phase) and artifact_present


def open_session(out_dir: str | Path, run_cfg: "RunConfig", run_id: str,
                 plan: list[Phase], *, resume: bool = False) -> Session:
    session_dir = Path(out_dir) / _SESSION_DIR
    session_dir.mkdir(parents=True, exist_ok=True)
    guard = run_cfg.agent.action_guard
    # The phase->allowed-actions + loop-back contract is static; (re)write it so an
    # external proposer can always read exactly what each phase permits (RFC §S3).
    atomic_write_json(session_dir / "allowed_actions.json", action_contract())
    if resume:
        load_session_manifest(session_dir)  # refuse (AqaError) when absent
        state = SessionState.load_or_init(session_dir, run_id, plan)
    else:
        write_session_manifest(session_dir, build_session_manifest(run_cfg, run_id))
        state = SessionState.new(run_id, plan)
        state.commit(session_dir)
        append_event(session_dir, "session_start", run_id=run_id,
                     plan=[p.value for p in plan])
    return Session(dir=session_dir, state=state,
                   guard_enabled=guard.enabled, guard_enforce=guard.enforce)
