"""Agent layer: fix-loop orchestration, tier ladder, verdict parsing, lessons.

Ports the Python-worthy orchestration from ``local_dispatch/agent`` -- the tier
escalation + verdict state machine (:mod:`fix_loop`), agent-output parsing
(:mod:`verdict`), deterministic drift signals (:mod:`drift`), lessons store
(:mod:`lessons`), tier ladder (:mod:`tiers`), prompt builder (:mod:`prompt`), and
the agent CLI backends (:mod:`backends`).  The thin in-container command execution
stays in shell.  Wired into the pipeline via :func:`aqa.agent.fix_loop.run_fix_loop`.
"""

from __future__ import annotations

from .backends import AgentUnavailable, run_agent_fix
from .fix_loop import FixLoop, FixLoopConfig, FixResult, PhaseResult, run_fix_loop
from .lessons import LessonStore
from .optimize import ModelAnalysis, analyze_model, run_agent_optimize
from .patch import PatchArea, PatchCapture, find_editable_areas
from .tiers import Tier, resolve_tiers
from .verdict import AgentAnalysis, Verdict, parse_verdict

__all__ = [
    "AgentAnalysis",
    "AgentUnavailable",
    "FixLoop",
    "FixLoopConfig",
    "FixResult",
    "LessonStore",
    "ModelAnalysis",
    "PatchArea",
    "PatchCapture",
    "PhaseResult",
    "Tier",
    "Verdict",
    "analyze_model",
    "find_editable_areas",
    "parse_verdict",
    "resolve_tiers",
    "run_agent_fix",
    "run_agent_optimize",
    "run_fix_loop",
]
