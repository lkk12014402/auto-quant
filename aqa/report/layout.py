"""Run-artifact layout contract: load ``layout.yaml`` and validate run dirs.

The layout file is the single source of truth for what an ``aqa run`` /
``aqa sweep`` output directory must contain.  :func:`inspect_run_dir` checks a
real directory against it and reports which expected entries are present /
missing, so owners (human or agent) can verify a run at a glance instead of
knowing the layout by heart.  Sink upload scope follows each entry's ``sink``
flag (weights and staged scratch are excluded by definition).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_SPEC_PATH = Path(__file__).with_name("layout.yaml")

_ALWAYS = "always"


@dataclass
class LayoutEntry:
    path: str
    kind: str
    when: str
    sink: bool
    desc: str


@dataclass
class InspectReport:
    run_dir: Path
    present: list[LayoutEntry] = field(default_factory=list)
    missing: list[LayoutEntry] = field(default_factory=list)
    not_applicable: list[LayoutEntry] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def load_layout(spec_path: Path | None = None) -> list[LayoutEntry]:
    data = yaml.safe_load((spec_path or _SPEC_PATH).read_text())
    return [LayoutEntry(path=e["path"], kind=e["kind"], when=e["when"],
                        sink=bool(e.get("sink", True)), desc=e.get("desc", ""))
            for e in data["layout"]]


def _conditions(run_dir: Path) -> set[str]:
    """Which ``when`` conditions hold for this run dir (best effort)."""
    conds: set[str] = set()
    state = ""
    try:
        state = json.loads((run_dir / "aqa_run.json").read_text()).get("state", "")
    except (OSError, json.JSONDecodeError):
        pass
    if state == "done":
        conds.add("done")
    for name, cond in (
        ("quantized", "quantize"), ("eval_thinking_off", "thinking_dual"),
        ("baseline", "baseline"), ("baseline_thinking_off", "thinking_dual"),
        ("gate.json", "gate"), ("patches", "fix_loop"),
        ("patch_snapshot", "patch_capture"), ("sweep_report.json", "sweep"),
        (".aqa_local_docker", "staged_docker"), (".aqa_local_venv", "staged_venv"),
    ):
        if (run_dir / name).exists():
            conds.add(cond)
    if any(p.is_dir() and (p / "run_meta.json").is_file()
           for p in run_dir.iterdir() if p.is_dir()):
        conds.add("local_fs_sink")
    return conds


def inspect_run_dir(run_dir: str | Path,
                    spec_path: Path | None = None) -> InspectReport:
    """Check ``run_dir`` against the layout contract.

    An entry counts as expected when its ``when`` is ``always`` or its
    condition already holds (detected from the dir itself — e.g. a sibling
    artifact proves the feature was on).  ``missing`` = expected but absent.
    """
    run_dir = Path(run_dir)
    report = InspectReport(run_dir=run_dir)
    conds = _conditions(run_dir)
    for entry in load_layout(spec_path):
        expected = entry.when == _ALWAYS or entry.when in conds
        if "<" in entry.path:
            # Variable-path entries (e.g. <artifact_name>/) can't be stat'ed
            # literally; the detected condition already proves their presence.
            if entry.when in conds:
                report.present.append(entry)
            elif expected:
                report.missing.append(entry)
            else:
                report.not_applicable.append(entry)
            continue
        exists = (run_dir / entry.path.rstrip("/")).exists()
        if exists:
            report.present.append(entry)
        elif expected:
            report.missing.append(entry)
        else:
            report.not_applicable.append(entry)
    return report


def format_report(report: InspectReport) -> str:
    lines = [f"run dir: {report.run_dir}", f"status: {'OK' if report.ok else 'INCOMPLETE'}"]
    for tag, entries in (("present", report.present),
                         ("MISSING", report.missing)):
        for e in entries:
            lines.append(f"  [{tag:<7}] {e.path:<28} ({e.when}) {e.desc.split('；')[0].split('——')[0]}")
    return "\n".join(lines)
