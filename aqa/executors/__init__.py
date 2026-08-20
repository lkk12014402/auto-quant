"""Executor protocol: where/how a run physically executes.

Mirrors NeMo Evaluator's ``Executor`` Protocol.  Dispatch is
``EXECUTORS.create(cfg.executor.type)`` -- no if/elif trees.  Machine access
details (docker base image, proxy, ssh creds via ``*_env``, jump chain, machine
catalog) all live in :class:`~aqa.config.ExecutorConfig`, so private Intel infra
can ship as a separate ``aqa-executor-intel`` plugin without touching the core.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable


class ProcessState(str, Enum):
    UNKNOWN = "unknown"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class RunResult:
    output_dir: Path
    state: ProcessState


@runtime_checkable
class Executor(Protocol):
    name: str

    def run(self, run_cfg, *, dry_run: bool = False, resume: bool = False) -> RunResult:
        ...

    def status(self, output_dir: Path) -> ProcessState:
        ...

    def stop(self, output_dir: Path) -> bool:
        ...

    @staticmethod
    def detect(output_dir: Path) -> bool:
        ...
