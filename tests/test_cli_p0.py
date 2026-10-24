"""P0 CLI contract: `aqa inspect` layout gate + honest `--resume` error.

Covers docs/rfc/0001-reproducible-session-state-machine.md P0:
 - a run dir can be checked against the artifact layout from the CLI, and
 - `--resume` fails loudly instead of silently doing a fresh run.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from aqa.cli import main


def _mk_run(d: Path, *, state: str = "done", entries: tuple[str, ...] = ()) -> Path:
    (d / "aqa_run.json").write_text(json.dumps({"model": "org/m", "state": state}))
    (d / "logs").mkdir(exist_ok=True)
    for rel in entries:
        p = d / rel.rstrip("/")
        if rel.endswith("/"):
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
    return d


def test_inspect_ok_run_exits_zero(tmp_path):
    d = _mk_run(tmp_path, entries=("run_manifest.json", "eval/accuracy.json"))
    r = CliRunner().invoke(main, ["inspect", str(d)])
    assert r.exit_code == 0, r.output
    assert "OK" in r.output


def test_inspect_incomplete_run_exits_nonzero(tmp_path):
    # done state but no run_manifest.json -> INCOMPLETE
    d = _mk_run(tmp_path, entries=())
    r = CliRunner().invoke(main, ["inspect", str(d)])
    assert r.exit_code != 0
    assert "INCOMPLETE" in r.output
    assert "run_manifest.json" in r.output


def test_run_resume_without_session_errors(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model: org/model\noutput:\n  dir: {tmp_path / 'out'}\n")
    r = CliRunner().invoke(main, ["run", str(cfg), "--resume"])
    assert r.exit_code != 0
    # refuse loudly instead of silently starting fresh
    assert "session" in (str(r.exception) + r.output).lower()
