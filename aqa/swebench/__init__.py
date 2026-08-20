"""SWE-bench Verified adapter (agentic, eval-centric).

Turns the released SWE-bench dataset into agentic tasks graded with the official
FAIL_TO_PASS / PASS_TO_PASS rule, reusing the E2 sandbox + E3 ReAct agent stack.
The heavy pieces (Docker containers, repo test commands) are injectable, so the
scoring logic is fully unit-tested without Docker; repo/version test specifics are
delegated to the upstream ``swebench`` package when it's installed.

Typical use::

    from aqa.swebench import load_swebench, run_swebench
    insts = load_swebench("swebench_verified.parquet", limit=20)
    agent_result, report = run_swebench(insts, client, out_dir="out/swe")
    print(report.resolved_rate)
"""

from __future__ import annotations

from .dataset import SWEBenchInstance, load_swebench
from .environment import SWEBenchEnvironment
from .report import (
    InstanceResult,
    SWEBenchRunReport,
    build_report,
    write_predictions,
    write_report,
)
from .run import run_swebench, run_swebench_async
from .spec import (
    ResolutionReport,
    build_instruction,
    compute_resolution,
    docker_image_for,
    parse_pytest_log,
    resolve_spec,
)

__all__ = [
    "InstanceResult",
    "ResolutionReport",
    "SWEBenchEnvironment",
    "SWEBenchInstance",
    "SWEBenchRunReport",
    "build_instruction",
    "build_report",
    "compute_resolution",
    "docker_image_for",
    "load_swebench",
    "parse_pytest_log",
    "resolve_spec",
    "run_swebench",
    "run_swebench_async",
    "write_predictions",
    "write_report",
]
