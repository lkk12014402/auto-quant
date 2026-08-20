"""Code-execution scorer -- the LiveCodeBench / HumanEval / MBPP / SciCode family.

The model under test writes code; grading = *run it against the benchmark's tests*.
This is the code-execution half of the cross-cutting scoring primitives (the other
being ``llm_judge``). Like the judge, the executor is **injected** (``CodeRunner``)
so the scorer is unit-testable offline, and the default runner reuses the existing
sandbox stack.

Safety note: :class:`LocalCodeRunner` executes generated code on the host (in a
throwaway temp dir). That is fine for trusted benchmarks/CI, but for untrusted
generations inject a container-backed runner (``DockerSandbox``) instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from ..sandbox.local import LocalSandbox
from .scoring import ScorerInput

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Return the first fenced ```python block, else the raw text (stripped)."""
    match = _FENCE_RE.search(text)
    return (match.group(1) if match else text).strip("\n")


@dataclass
class CodeExecResult:
    passed: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class CodeRunner(Protocol):
    """Runs one program to completion and reports whether it passed."""

    def run(self, program: str, *, timeout: float) -> CodeExecResult:
        ...


class LocalCodeRunner:
    """Runs a single program in a throwaway :class:`LocalSandbox` on the host."""

    def __init__(self, *, interpreter: str = "python3", filename: str = "solution.py"):
        self.interpreter = interpreter
        self.filename = filename

    def run(self, program: str, *, timeout: float) -> CodeExecResult:
        # allow_unsafe: the *program* is arbitrary benchmark code; the guard only
        # inspects the launch command (`python3 solution.py`), which is benign.
        sandbox = LocalSandbox(allow_unsafe=True, default_timeout=timeout)
        with sandbox:
            sandbox.write_file(self.filename, program)
            result = sandbox.exec(f"{self.interpreter} {self.filename}", timeout=timeout)
        return CodeExecResult(
            passed=result.ok,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
        )


def build_program(sample: dict[str, Any], completion: str, spec: Any) -> str:
    """Assemble the full program: [preamble] + [prompt?] + completion + test + check."""
    include_prompt = getattr(spec, "include_prompt", False)
    prompt_field = getattr(spec, "prompt_field", "prompt")
    test_field = getattr(spec, "test_field", "test")
    entry_field = getattr(spec, "entry_point_field", "entry_point")
    check_template = getattr(spec, "check_template", "\ncheck({entry_point})\n")
    preamble = getattr(spec, "preamble", None)

    parts: list[str] = []
    if preamble:
        parts.append(preamble)
    prefix = str(sample.get(prompt_field, "")) if include_prompt else ""
    parts.append(prefix + completion if prefix else completion)
    test = str(sample.get(test_field, ""))
    if test:
        parts.append(test)
    entry = str(sample.get(entry_field, ""))
    if check_template and entry:
        parts.append(check_template.format(entry_point=entry))
    return "\n\n".join(p for p in parts if p)


def build_code_exec_scorer(spec: Any, runner: CodeRunner | None = None) -> Any:
    """Build a ``code_exec`` scorer callable from a :class:`CodeExecSpec`-like object.

    ``runner`` is injected in tests; when omitted a :class:`LocalCodeRunner` is
    built lazily from the spec.
    """
    extract = getattr(spec, "extract", True) if spec is not None else True
    timeout = float(getattr(spec, "timeout_s", 30.0) or 30.0) if spec is not None else 30.0

    def _runner() -> CodeRunner:
        nonlocal runner
        if runner is None:
            runner = LocalCodeRunner(
                interpreter=getattr(spec, "interpreter", None) or "python3",
            )
        return runner

    def code_exec(sample: ScorerInput) -> dict[str, Any]:
        completion = extract_code(sample.prediction) if extract else sample.prediction
        raw = sample.extra.get("sample") or {}
        program = build_program(raw, completion, spec)
        result = _runner().run(program, timeout=timeout)
        return {
            "score": 1.0 if result.passed else 0.0,
            "metric": "code_exec",
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "stderr": result.stderr[:2000],
        }

    return code_exec


def _default_code_exec(sample: ScorerInput) -> dict[str, Any]:
    """Zero-config entry-point target: reads a CodeExecSpec off ``extra``."""
    return build_code_exec_scorer(sample.extra.get("code_exec_spec"))(sample)
