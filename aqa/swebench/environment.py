"""SWE-bench environment: drive an agent inside a released eval container and
grade its edits with the official FAIL_TO_PASS / PASS_TO_PASS rule.

:class:`SWEBenchEnvironment` implements the same async ``reset``/``step``/``close``
+ ``is_solved`` protocol the ReAct :class:`~aqa.agent_eval.react.ToolUseSolver`
drives, so SWE-bench reuses the whole agentic-eval stack.  What's SWE-bench
specific lives here:

* ``reset`` puts ``/testbed`` back to ``base_commit`` (clean), so the diff we
  capture is exactly the agent's work.
* ``is_solved`` captures the model patch (``git diff``), applies the gold
  ``test_patch``, runs the (official or generic) test command, parses the log,
  and applies :func:`~aqa.swebench.spec.compute_resolution`.  The captured patch
  and grading report are retained for the prediction/report writers.

Every blocking sandbox call is dispatched with ``asyncio.to_thread`` so many
instances grade concurrently under the async loop.  All Docker work goes through
the injectable :class:`~aqa.sandbox.base.Sandbox`, so the grading logic is unit
-tested with a fake sandbox — no Docker, no real repos.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..agent_eval.environment import Observation, _truncate
from ..logging_util import get_logger
from .spec import (
    apply_test_patch_cmd,
    capture_patch_cmd,
    compute_resolution,
    reset_repo_cmd,
    resolve_spec,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from ..sandbox.base import Sandbox
    from .dataset import SWEBenchInstance
    from .spec import ResolutionReport, TestSpec

_log = get_logger("swebench.env")

TEST_PATCH_PATH = "/tmp/test.patch"


class SWEBenchEnvironment:
    """Agentic environment for one SWE-bench instance (graded on real tests)."""

    def __init__(
        self,
        instance: SWEBenchInstance,
        sandbox: Sandbox,
        *,
        spec_provider: Callable[[SWEBenchInstance], TestSpec | None] | None = None,
        exec_timeout: float = 900.0,
        test_timeout: float = 1800.0,
        max_output_chars: int = 4000,
    ):
        self.instance = instance
        self.sandbox = sandbox
        self.spec = resolve_spec(instance, spec_provider)
        self.exec_timeout = exec_timeout
        self.test_timeout = test_timeout
        self.max_output_chars = max_output_chars
        # Filled by is_solved (consumed by the prediction/report writers).
        self.model_patch: str = ""
        self.report: ResolutionReport | None = None

    async def reset(self) -> Observation:
        await asyncio.to_thread(self.sandbox.start)
        await asyncio.to_thread(
            self.sandbox.exec,
            reset_repo_cmd(self.instance.base_commit),
            timeout=self.exec_timeout,
        )
        return Observation(text="", info={"instance_id": self.instance.instance_id})

    async def step(self, action: str) -> Observation:
        res = await asyncio.to_thread(
            self.sandbox.exec, action, timeout=self.exec_timeout
        )
        return Observation(
            text=_truncate(res.output, self.max_output_chars),
            info={"returncode": res.returncode, "timed_out": res.timed_out},
        )

    async def is_solved(self) -> bool:
        report = await asyncio.to_thread(self._grade)
        self.report = report
        return report.resolved

    def _grade(self) -> ResolutionReport:
        """Capture the patch, apply gold tests, run + parse, compute resolution.

        Runs on a worker thread (all-synchronous sandbox calls).
        """
        inst = self.instance
        # 1) Capture the agent's fix as a patch (excludes test files).
        self.model_patch = self.sandbox.exec(
            capture_patch_cmd(), timeout=self.exec_timeout).stdout

        # 2) Overlay the gold test patch so the *hidden* tests exist.
        if inst.test_patch.strip():
            self.sandbox.write_file(TEST_PATCH_PATH, inst.test_patch)
            self.sandbox.exec(apply_test_patch_cmd(), timeout=self.exec_timeout)

        # 3) Run the target tests and parse per-test statuses.
        run = self.sandbox.exec(self.spec.test_cmd, timeout=self.test_timeout)
        status_map = self.spec.log_parser(run.output)

        # 4) Apply the official resolution rule.
        report = compute_resolution(inst.fail_to_pass, inst.pass_to_pass, status_map)
        _log.info(
            "graded %s: resolved=%s (f2p %s, p2p %s)",
            inst.instance_id, report.resolved, report.f2p_success, report.p2p_success,
        )
        return report

    async def close(self) -> None:
        await asyncio.to_thread(self.sandbox.close)
