"""Sandbox abstraction: where a tool-use / terminal task's commands execute.

Agentic and terminal benchmarks (E3/E4) drive the model to *act* -- run shell
commands, edit files, inspect output -- inside an isolated workspace.  A
:class:`Sandbox` is that workspace: a uniform ``exec`` / ``write_file`` /
``read_file`` surface over an execution backend.

Two backends ship in core:

  * ``local``  -- :class:`~aqa.sandbox.local.LocalSandbox`, a scratch directory on
    the host (with an unsafe-command guard, since it is *not* isolated).
  * ``docker`` -- :class:`~aqa.sandbox.docker.DockerSandbox`, a throwaway
    container (true isolation; the model may run anything).

Heavier infra (ECS / Slurm / Apptainer) ships as ``aqa-sandbox-*`` plugins via
the ``aqa.sandboxes`` entry-point group -- no fork of core required.  Dispatch is
:func:`create_sandbox` -> ``SANDBOXES.create(...)``, no if/elif trees.
"""

from __future__ import annotations

from typing import Any

from ..registry import SANDBOXES
from .base import BaseSandbox, ExecResult, Sandbox, SandboxError, is_unsafe_command
from .docker import DockerSandbox
from .local import LocalSandbox


def create_sandbox(kind: str = "local", **options: Any) -> Sandbox:
    """Instantiate a registered sandbox backend by name.

    ``create_sandbox("docker", image="python:3.12")`` mirrors
    ``EXECUTORS.create(...)`` -- builtin ``local``/``docker`` plus any installed
    ``aqa.sandboxes`` plugin.
    """
    return SANDBOXES.create(kind, **options)


__all__ = [
    "BaseSandbox",
    "DockerSandbox",
    "ExecResult",
    "LocalSandbox",
    "Sandbox",
    "SandboxError",
    "create_sandbox",
    "is_unsafe_command",
]
