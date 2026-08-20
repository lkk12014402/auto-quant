"""SWE-bench spec + scoring core (harness-independent).

This module owns the *pure* pieces of SWE-bench evaluation — the parts that must
be correct and are therefore unit-tested without Docker:

* :func:`docker_image_for` — resolve the per-instance evaluation image (the
  released ``swebench/sweb.eval.<arch>.<id>`` containers, repo checked out at
  ``/testbed``).
* :func:`build_instruction` — the prompt shown to the agent.
* patch capture / test-application command builders (git plumbing strings).
* :func:`parse_pytest_log` — map a pytest ``-rA`` log to per-test statuses.
* :func:`compute_resolution` — the official FAIL_TO_PASS / PASS_TO_PASS rule that
  decides whether an instance is *resolved*.

Repo/version-specific *test commands* are delegated to the official ``swebench``
package when installed (via an injectable ``spec_provider``); otherwise a generic
pytest spec is used so the adapter still runs end-to-end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from .dataset import SWEBenchInstance

# Where the released images check the repository out.
TESTBED = "/testbed"

# --------------------------------------------------------------------------
# Test status vocabulary (subset of the official SWE-bench statuses).
# --------------------------------------------------------------------------
PASSED = "PASSED"
FAILED = "FAILED"
ERROR = "ERROR"
SKIPPED = "SKIPPED"
# A test named in the spec but absent from the log never counts as a pass.
MISSING = "MISSING"


# --------------------------------------------------------------------------
# Image resolution
# --------------------------------------------------------------------------
def image_key(instance_id: str) -> str:
    """The registry-safe tag fragment for an instance.

    The shared SWE-bench images lower-case the id and replace ``__`` with
    ``_1776_`` (Docker tags can't contain ``//`` when namespaced), e.g.
    ``astropy__astropy-12907`` -> ``astropy_1776_astropy-12907``.
    """
    return instance_id.lower().replace("__", "_1776_")


def docker_image_for(
    instance: SWEBenchInstance,
    *,
    namespace: str = "swebench",
    arch: str = "x86_64",
    tag: str = "latest",
) -> str:
    """Resolve the evaluation image for ``instance`` (overridable per field)."""
    return f"{namespace}/sweb.eval.{arch}.{image_key(instance.instance_id)}:{tag}"


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
_INSTRUCTION = """\
You are working in a checked-out clone of the `{repo}` repository at {testbed}.

Resolve the following issue by editing the repository's **source** code. Do not
edit the test files — hidden tests will be run against your changes.

<issue>
{problem}
</issue>

When you are confident the issue is fixed, submit your final answer. Your changes
in {testbed} will be captured (via `git diff`) and graded against the project's
test suite.
"""


def build_instruction(instance: SWEBenchInstance) -> str:
    """The task prompt: the issue text + where/how the work is graded."""
    return _INSTRUCTION.format(
        repo=instance.repo,
        testbed=TESTBED,
        problem=instance.problem_statement.strip() or "(no description provided)",
    )


# --------------------------------------------------------------------------
# Git plumbing (command strings run inside the sandbox)
# --------------------------------------------------------------------------
def reset_repo_cmd(base_commit: str, testbed: str = TESTBED) -> str:
    """Return the repo to a clean ``base_commit`` state before the agent starts."""
    return (
        f"cd {testbed} && git reset --hard {base_commit} "
        f"&& git clean -fdqx -e '*.egg-info'"
    )


def capture_patch_cmd(testbed: str = TESTBED) -> str:
    """Capture the agent's edits as a unified diff (the model patch).

    Excludes changes under common test directories so the captured patch is a
    *fix*, mirroring the official prediction shape (tests come from test_patch).
    """
    return (
        f"cd {testbed} && git add -A "
        f"&& git diff --cached -- . ':(exclude)**/tests/**' ':(exclude)**/test/**' "
        f"':(exclude)**/*_test.py' ':(exclude)**/test_*.py'"
    )


def apply_test_patch_cmd(testbed: str = TESTBED) -> str:
    """Apply the gold ``test_patch`` (written to /tmp/test.patch) onto the tree."""
    return (
        f"cd {testbed} && git apply -v /tmp/test.patch "
        f"|| patch --batch --fuzz=5 -p1 -i /tmp/test.patch"
    )


# --------------------------------------------------------------------------
# Test spec (command + parser); official-delegating with a generic fallback
# --------------------------------------------------------------------------
@dataclass
class TestSpec:
    """How to run + parse the tests for one instance."""

    test_cmd: str
    log_parser: Callable[[str], dict[str, str]]


def generic_pytest_spec(instance: SWEBenchInstance) -> TestSpec:
    """A repo-agnostic pytest spec covering the instance's target tests.

    Used when the official ``swebench`` spec map is unavailable.  Runs exactly the
    FAIL_TO_PASS + PASS_TO_PASS node ids with ``-rA`` so every outcome is logged.
    """
    tests = list(dict.fromkeys([*instance.fail_to_pass, *instance.pass_to_pass]))
    joined = " ".join(_shq(t) for t in tests) if tests else ""
    cmd = f"cd {TESTBED} && python -m pytest -rA -p no:cacheprovider {joined}".rstrip()
    return TestSpec(test_cmd=cmd, log_parser=parse_pytest_log)


def official_spec_provider(instance: SWEBenchInstance) -> TestSpec | None:
    """Build a :class:`TestSpec` from the installed ``swebench`` package, if any.

    Returns ``None`` when ``swebench`` is not importable so the caller can fall
    back to :func:`generic_pytest_spec`.
    """
    try:  # pragma: no cover - exercised only when swebench is installed
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    except Exception:  # noqa: BLE001 - any import/attr issue -> fall back
        return None
    try:  # pragma: no cover
        specs = MAP_REPO_VERSION_TO_SPECS[instance.repo][instance.version]
        parser = MAP_REPO_TO_PARSER[instance.repo]
    except Exception:  # noqa: BLE001
        return None
    tests = " ".join(_shq(t) for t in [*instance.fail_to_pass, *instance.pass_to_pass])
    test_cmd = f"cd {TESTBED} && {specs['test_cmd']} {tests}".rstrip()
    return TestSpec(test_cmd=test_cmd, log_parser=lambda log: dict(parser(log)))


def resolve_spec(
    instance: SWEBenchInstance,
    spec_provider: Callable[[SWEBenchInstance], TestSpec | None] | None = None,
) -> TestSpec:
    """Pick the official spec when available, else the generic pytest spec."""
    provider = spec_provider or official_spec_provider
    return provider(instance) or generic_pytest_spec(instance)


def _shq(value: str) -> str:
    import shlex
    return shlex.quote(value)


# --------------------------------------------------------------------------
# Log parsing (generic pytest ``-rA`` short summary)
# --------------------------------------------------------------------------
_PYTEST_LINE = re.compile(
    r"^(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b\s+(?P<test>\S+)"
)
_STATUS_MAP = {
    "PASSED": PASSED,
    "XFAIL": PASSED,   # expected failure that failed as expected -> a pass
    "XPASS": PASSED,
    "FAILED": FAILED,
    "ERROR": ERROR,
    "SKIPPED": SKIPPED,
}


def parse_pytest_log(log: str) -> dict[str, str]:
    """Map a pytest ``-rA`` log to ``{test_node_id: STATUS}``."""
    results: dict[str, str] = {}
    for line in log.splitlines():
        m = _PYTEST_LINE.match(line.strip())
        if not m:
            continue
        results[m.group("test")] = _STATUS_MAP.get(m.group("status"), FAILED)
    return results


# --------------------------------------------------------------------------
# Resolution rule (official FAIL_TO_PASS / PASS_TO_PASS semantics)
# --------------------------------------------------------------------------
@dataclass
class ResolutionReport:
    """Per-instance grading detail."""

    resolved: bool
    fail_to_pass: dict[str, str] = field(default_factory=dict)   # test -> status
    pass_to_pass: dict[str, str] = field(default_factory=dict)
    f2p_success: bool = False
    p2p_success: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "resolved": self.resolved,
            "f2p_success": self.f2p_success,
            "p2p_success": self.p2p_success,
            "FAIL_TO_PASS": self.fail_to_pass,
            "PASS_TO_PASS": self.pass_to_pass,
        }


def compute_resolution(
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    status_map: dict[str, str],
) -> ResolutionReport:
    """Apply the official rule: an instance is *resolved* iff **every**
    FAIL_TO_PASS **and every** PASS_TO_PASS test is ``PASSED`` in the post-patch
    log.  A test absent from the log counts as ``MISSING`` (never a pass).
    """
    f2p = {t: status_map.get(t, MISSING) for t in fail_to_pass}
    p2p = {t: status_map.get(t, MISSING) for t in pass_to_pass}
    f2p_ok = all(s == PASSED for s in f2p.values()) if f2p else True
    p2p_ok = all(s == PASSED for s in p2p.values()) if p2p else True
    # A degenerate instance with no FAIL_TO_PASS tests can't be "resolved".
    resolved = bool(fail_to_pass) and f2p_ok and p2p_ok
    return ResolutionReport(
        resolved=resolved, fail_to_pass=f2p, pass_to_pass=p2p,
        f2p_success=f2p_ok, p2p_success=p2p_ok,
    )
