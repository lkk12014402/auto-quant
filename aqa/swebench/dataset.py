"""SWE-bench dataset loading.

A SWE-bench instance is a real GitHub issue + repo snapshot: the agent must edit
the repository so the hidden tests pass.  This module turns the released dataset
(a parquet shard such as ``swebench_verified.parquet``, or a JSON/JSONL export)
into typed :class:`SWEBenchInstance` records, with filtering by id / split / limit.

The heavy ``pyarrow`` dependency is imported lazily (only parquet needs it), so
core stays light and JSON/JSONL loading works with no extra install — which is
also what the unit tests use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_util import AqaError

# Columns of the released SWE-bench(-Verified) dataset we consume.
_STR_FIELDS = (
    "repo", "instance_id", "base_commit", "patch", "test_patch",
    "problem_statement", "hints_text", "created_at", "version",
    "environment_setup_commit", "difficulty",
)
_LIST_FIELDS = ("FAIL_TO_PASS", "PASS_TO_PASS")


def _as_list(value: Any) -> list[str]:
    """Normalise a FAIL_TO_PASS / PASS_TO_PASS cell into a list of test ids.

    The dataset stores these as a JSON-encoded string, but exports/tests may hand
    us an actual list; accept both.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AqaError(f"could not parse test list: {text[:80]!r}") from exc
    if not isinstance(parsed, list):
        raise AqaError(f"expected a list of tests, got {type(parsed).__name__}")
    return [str(v) for v in parsed]


@dataclass
class SWEBenchInstance:
    """One SWE-bench task instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str = ""
    patch: str = ""                       # the gold patch (never shown to the agent)
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    version: str = ""
    environment_setup_commit: str = ""
    hints_text: str = ""
    created_at: str = ""
    difficulty: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SWEBenchInstance:
        if not row.get("instance_id"):
            raise AqaError("SWE-bench row is missing 'instance_id'")
        if not row.get("repo") or not row.get("base_commit"):
            raise AqaError(
                f"SWE-bench instance {row.get('instance_id')!r} missing repo/base_commit")
        return cls(
            instance_id=str(row["instance_id"]),
            repo=str(row["repo"]),
            base_commit=str(row["base_commit"]),
            problem_statement=str(row.get("problem_statement", "")),
            test_patch=str(row.get("test_patch", "") or ""),
            patch=str(row.get("patch", "") or ""),
            fail_to_pass=_as_list(row.get("FAIL_TO_PASS")),
            pass_to_pass=_as_list(row.get("PASS_TO_PASS")),
            version=str(row.get("version", "") or ""),
            environment_setup_commit=str(row.get("environment_setup_commit", "") or ""),
            hints_text=str(row.get("hints_text", "") or ""),
            created_at=str(row.get("created_at", "") or ""),
            difficulty=str(row.get("difficulty", "") or ""),
        )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "instances" in data:
            data = data["instances"]
        if not isinstance(data, list):
            raise AqaError(f"{path}: expected a JSON list of instances")
        return data
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dep
            raise AqaError(
                "loading a .parquet SWE-bench dataset needs `pyarrow` "
                "(pip install 'autoquant-agent[swebench]')"
            ) from exc
        return pq.read_table(str(path)).to_pylist()
    raise AqaError(f"unsupported SWE-bench dataset format: {path.suffix!r}")


def load_swebench(
    path: str | Path,
    *,
    instance_ids: list[str] | None = None,
    limit: int | None = None,
    difficulty: str | None = None,
) -> list[SWEBenchInstance]:
    """Load SWE-bench instances from a parquet / json / jsonl file.

    * ``instance_ids`` -- keep only these ids (preserving dataset order).
    * ``difficulty``   -- keep only rows whose ``difficulty`` matches.
    * ``limit``        -- cap the number returned (after filtering).
    """
    p = Path(path)
    if not p.exists():
        raise AqaError(f"SWE-bench dataset not found: {p}")
    wanted = set(instance_ids) if instance_ids else None
    instances: list[SWEBenchInstance] = []
    for row in _read_rows(p):
        inst = SWEBenchInstance.from_row(row)
        if wanted is not None and inst.instance_id not in wanted:
            continue
        if difficulty is not None and inst.difficulty != difficulty:
            continue
        instances.append(inst)
    if wanted is not None:
        missing = wanted - {i.instance_id for i in instances}
        if missing:
            raise AqaError(f"instance_ids not found in dataset: {sorted(missing)}")
    if limit is not None:
        instances = instances[:limit]
    return instances
