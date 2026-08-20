"""Terminal-benchmark task definitions (E4).

A terminal task is a self-contained spec — *instruction*, optional seed *files*
and *setup* commands, and a *verifier* — that layers on the E2 sandbox + E3
tool-use agent.  It mirrors the shape of terminal-bench / SWE-style tasks:

    id: hello-file
    instruction: "Create /workspace/hello.txt containing 'hello world'."
    setup: ["mkdir -p /workspace"]
    files:
      note.md: "leave hello.txt in /workspace"
    verify:
      files:
        - path: hello.txt
          equals: "hello world"
    max_turns: 8

The verifier runs *after* the agent finishes (or exhausts its turn budget): a
shell ``command`` that must exit 0, and/or declarative ``files`` checks.  Both are
combined with AND, so a task passes only when every check passes.  This keeps
authoring dependency-free (no pytest harness required) yet expressive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..logging_util import AqaError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from ..sandbox.base import Sandbox


@dataclass
class FileCheck:
    """A declarative assertion about one file in the sandbox."""

    path: str
    equals: str | None = None       # exact content (trailing newline forgiven)
    contains: str | None = None     # substring must be present
    exists: bool = True             # file must be readable

    def check(self, sandbox: Sandbox) -> bool:
        from ..sandbox.base import SandboxError

        try:
            content = sandbox.read_file(self.path)
        except SandboxError:
            return not self.exists
        if not self.exists:
            return False
        if self.equals is not None and content.rstrip("\n") != self.equals.rstrip("\n"):
            return False
        return not (self.contains is not None and self.contains not in content)


@dataclass
class Verifier:
    """How a task's success is decided (all checks AND-combined)."""

    command: str | None = None
    files: list[FileCheck] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Verifier:
        data = data or {}
        files = [
            FileCheck(**fc) if isinstance(fc, dict) else FileCheck(path=str(fc))
            for fc in data.get("files", [])
        ]
        return cls(command=data.get("command"), files=files)

    def build(self) -> Callable[[Sandbox], bool]:
        command = self.command
        checks = list(self.files)

        def verify(sandbox: Sandbox) -> bool:
            if command is not None and not sandbox.exec(command).ok:
                return False
            return all(fc.check(sandbox) for fc in checks)

        return verify


@dataclass
class TerminalTask:
    """One terminal-benchmark task."""

    id: str
    instruction: str
    setup: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    verify: Verifier = field(default_factory=Verifier)
    image: str | None = None
    sandbox: str | None = None      # override backend: "local" / "docker"
    max_turns: int = 15
    timeout: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> TerminalTask:
        if "id" not in data or "instruction" not in data:
            raise AqaError("terminal task requires 'id' and 'instruction'")
        return cls(
            id=str(data["id"]),
            instruction=str(data["instruction"]),
            setup=list(data.get("setup", [])),
            files=_load_files(data.get("files", {}), base_dir),
            verify=Verifier.from_dict(data.get("verify")),
            image=data.get("image"),
            sandbox=data.get("sandbox"),
            max_turns=int(data.get("max_turns", 15)),
            timeout=float(data.get("timeout", 60.0)),
            metadata=dict(data.get("metadata", {})),
        )


def _load_files(spec: dict[str, Any], base_dir: Path | None) -> dict[str, str]:
    """Resolve a ``files`` mapping: literal strings, or ``{path: <relpath>}``."""
    out: dict[str, str] = {}
    for dest, value in spec.items():
        if isinstance(value, dict) and "path" in value:
            if base_dir is None:
                raise AqaError(f"file ref for {dest!r} needs a base directory")
            out[dest] = (base_dir / value["path"]).read_text(encoding="utf-8")
        else:
            out[dest] = str(value)
    return out


def load_terminal_task(path: str | Path) -> TerminalTask:
    """Load a single task from a YAML file (or a dir containing ``task.yaml``)."""
    p = Path(path)
    if p.is_dir():
        p = p / "task.yaml"
    if not p.exists():
        raise AqaError(f"terminal task file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    return TerminalTask.from_dict(data, base_dir=p.parent)


def load_terminal_tasks(path: str | Path) -> list[TerminalTask]:
    """Load a suite: a single YAML file, or a directory of task YAMLs / task dirs."""
    p = Path(path)
    if p.is_file():
        raw = yaml.safe_load(p.read_text()) or {}
        if isinstance(raw, list):
            return [TerminalTask.from_dict(d, base_dir=p.parent) for d in raw]
        if "tasks" in raw:
            return [TerminalTask.from_dict(d, base_dir=p.parent) for d in raw["tasks"]]
        return [TerminalTask.from_dict(raw, base_dir=p.parent)]
    if not p.is_dir():
        raise AqaError(f"terminal task path not found: {p}")
    tasks: list[TerminalTask] = []
    for entry in sorted(p.iterdir()):
        is_task_dir = entry.is_dir() and (entry / "task.yaml").exists()
        is_task_yaml = entry.suffix in (".yaml", ".yml") and entry.name != "suite.yaml"
        if is_task_dir or is_task_yaml:
            tasks.append(load_terminal_task(entry))
    if not tasks:
        raise AqaError(f"no terminal tasks found under {p}")
    return tasks


def builtin_task_pack() -> list[TerminalTask]:
    """Load the small example task pack shipped with the package."""
    return load_terminal_tasks(Path(__file__).resolve().parent / "tasks")
