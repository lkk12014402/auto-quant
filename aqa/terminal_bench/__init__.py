"""Terminal-benchmark task definitions + runner (E4).

Concrete, self-contained task specs (instruction + seed files + setup + verifier)
that run on the E2 sandbox through the E3 tool-use agent.  Load a suite with
:func:`load_terminal_tasks` (or the shipped :func:`builtin_task_pack`) and score a
served model with :func:`run_terminal_bench`.
"""

from __future__ import annotations

from .run import build_env_factory, run_terminal_bench, to_agent_task
from .task import (
    FileCheck,
    TerminalTask,
    Verifier,
    builtin_task_pack,
    load_terminal_task,
    load_terminal_tasks,
)

__all__ = [
    "FileCheck",
    "TerminalTask",
    "Verifier",
    "build_env_factory",
    "builtin_task_pack",
    "load_terminal_task",
    "load_terminal_tasks",
    "run_terminal_bench",
    "to_agent_task",
]
