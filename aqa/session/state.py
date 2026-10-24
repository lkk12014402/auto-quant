"""The session state machine (``state.json``).

Models aqa's quantize run as an explicit phase sequence with per-phase status,
plus the running baseline / current-best / decision log that today only lives in
transcripts and scattered logs.  Shaped after Hyperloom's
``orchestrator/state/shared_state.py:SharedState`` (``load_or_init`` /
``state_path`` / ``from_dict`` / ``to_dict``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ._io import atomic_write_json, now_iso

SCHEMA_VERSION = 1
_STATE_FILE = "state.json"


class Phase(str, Enum):
    """The pipeline phase sequence (superset; ``build_phase_plan`` trims per-run).

    ``ANALYZE/ENV/PLAN`` map to the AGENTS.md owner hard-flow steps that run
    before ``aqa run``; ``BASELINE..CLOSE`` are the runtime phases ``run_pipeline``
    actually executes and can resume.
    """

    ANALYZE = "ANALYZE"
    ENV = "ENV"
    PLAN = "PLAN"
    BASELINE = "BASELINE"
    QUANT = "QUANT"
    EVAL = "EVAL"
    GATE = "GATE"
    VERIFY = "VERIFY"
    PUBLISH = "PUBLISH"
    CLOSE = "CLOSE"


class Status(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


def state_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / _STATE_FILE


@dataclass
class SessionState:
    """Mutable run progress + decisions, persisted to ``session/state.json``."""

    schema_version: int
    run_id: str
    phase: str                                   # current phase (name)
    phases: dict[str, str]                       # phase name -> Status value
    baseline: dict | None = None                 # {mean_score, per_task}
    current_best: dict | None = None             # {scheme, mean_score, size_gb?}
    decisions: list[dict] = field(default_factory=list)
    gate: dict | None = None
    created_at: str = ""
    updated_at: str = ""

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def new(cls, run_id: str, plan: list[Phase]) -> "SessionState":
        now = now_iso()
        return cls(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            phase=plan[0].value if plan else "",
            phases={p.value: Status.PENDING.value for p in plan},
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            run_id=data.get("run_id", ""),
            phase=data.get("phase", ""),
            phases=dict(data.get("phases", {})),
            baseline=data.get("baseline"),
            current_best=data.get("current_best"),
            decisions=list(data.get("decisions", [])),
            gate=data.get("gate"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @classmethod
    def load_or_init(cls, session_dir: str | Path, run_id: str,
                     plan: list[Phase]) -> "SessionState":
        path = state_path(session_dir)
        if path.exists():
            import json
            return cls.from_dict(json.loads(path.read_text()))
        return cls.new(run_id, plan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "phase": self.phase,
            "phases": dict(self.phases),
            "baseline": self.baseline,
            "current_best": self.current_best,
            "decisions": list(self.decisions),
            "gate": self.gate,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def commit(self, session_dir: str | Path) -> None:
        self.updated_at = now_iso()
        atomic_write_json(state_path(session_dir), self.to_dict())

    # ── transitions ─────────────────────────────────────────────────────────
    def status_of(self, phase: Phase) -> str:
        return self.phases.get(phase.value, Status.PENDING.value)

    def is_done(self, phase: Phase) -> bool:
        return self.status_of(phase) == Status.DONE.value

    def _set(self, phase: Phase, status: Status) -> None:
        self.phases[phase.value] = status.value
        self.updated_at = now_iso()

    def enter(self, phase: Phase) -> None:
        self.phase = phase.value
        self._set(phase, Status.RUNNING)

    def done(self, phase: Phase) -> None:
        self._set(phase, Status.DONE)

    def skip(self, phase: Phase) -> None:
        self._set(phase, Status.SKIPPED)

    def fail(self, phase: Phase) -> None:
        self._set(phase, Status.FAILED)

    def record_decision(self, phase: Phase, what: str, why: str, ref: str = "") -> None:
        self.decisions.append({"phase": phase.value, "what": what, "why": why, "ref": ref})
        self.updated_at = now_iso()

    # ── prompt re-seeding (RFC §3.5 anti-drift) ──────────────────────────────
    def prompt_seed(self, *, max_decisions: int = 6) -> str:
        """Render the mission-critical state as a compact prompt block.

        Used to re-seed optimize/fix prompts from ``state.json`` rather than from a
        drift-prone transcript: current phase, the baseline/current-best scores the
        run must not regress, and the decisions already made (newest first).
        """
        lines = [f"- run: {self.run_id}  |  current phase: {self.phase or '?'}"]
        if self.baseline:
            lines.append(f"- baseline (do NOT regress below this): {_fmt_scores(self.baseline)}")
        if self.current_best:
            lines.append(f"- current best so far: {_fmt_scores(self.current_best)}")
        if self.gate:
            verdict = self.gate.get("verdict")
            if verdict:
                lines.append(f"- last gate verdict: {verdict}")
        if self.decisions:
            lines.append("- decisions already made (honor these):")
            for d in reversed(self.decisions[-max_decisions:]):
                what = str(d.get("what", "")).strip()
                why = str(d.get("why", "")).strip()
                phase = str(d.get("phase", "")).strip()
                tail = f" — {why}" if why else ""
                lines.append(f"    - [{phase}] {what}{tail}"[:200])
        return "\n".join(lines)


def _fmt_scores(summary: dict) -> str:
    """One-line 'k=v' render of a score summary dict (mean_score first)."""
    if not isinstance(summary, dict):
        return str(summary)
    parts: list[str] = []
    for key in ("scheme", "mean_score"):
        if summary.get(key) is not None:
            parts.append(f"{key}={summary[key]}")
    for k, v in summary.items():
        if k not in ("scheme", "mean_score") and v is not None:
            parts.append(f"{k}={v}")
    return ", ".join(parts) or "{}"
