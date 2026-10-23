"""Tests for the run-artifact layout contract."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from aqa.cli import main
from aqa.report.layout import inspect_run_dir, load_layout


def _mk_run(tmp_path: Path, *, state: str = "done",
            entries: tuple[str, ...] = ()) -> Path:
    (tmp_path / "aqa_run.json").write_text(json.dumps({"model": "org/m", "state": state}))
    (tmp_path / "logs").mkdir(exist_ok=True)  # executor logs exist for any real run
    for rel in entries:
        p = tmp_path / rel.rstrip("/")
        if rel.endswith("/"):
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
    return tmp_path


def test_spec_loads():
    entries = load_layout()
    assert entries and all(e.path and e.when and e.kind in ("file", "dir") for e in entries)
    assert any(e.path == "quantized/" and e.sink is False for e in entries)


def test_minimal_done_run_is_ok(tmp_path):
    d = _mk_run(tmp_path, entries=("eval/accuracy.json",))
    # 只建 aqa_run.json 时 done 状态会缺 run_manifest.json 与 eval/accuracy.json
    report = inspect_run_dir(d)
    missing = {e.path for e in report.missing}
    assert "run_manifest.json" in missing

    _mk_run(tmp_path, entries=("run_manifest.json",))
    report2 = inspect_run_dir(d)
    assert report2.ok, [e.path for e in report2.missing]


def test_quantize_run_requires_quantized_and_command(tmp_path):
    d = _mk_run(tmp_path, entries=(
        "run_manifest.json", "eval/accuracy.json", "quantized/",
        "quantized/quantize.log"))
    report = inspect_run_dir(d)
    # quantized/ 存在 => quantize 条件成立 => .quant_command.json 变为期望项
    assert ".quant_command.json" in {e.path for e in report.missing}
    _mk_run(tmp_path, entries=(".quant_command.json",))
    assert inspect_run_dir(d).ok


def test_thinking_dual_detected(tmp_path):
    d = _mk_run(tmp_path, entries=(
        "run_manifest.json", "eval/accuracy.json",
        "eval_thinking_off/accuracy.json", "baseline/accuracy.json",
        "baseline_thinking_off/accuracy.json"))
    report = inspect_run_dir(d)
    assert report.ok
    assert any(e.path == "baseline_thinking_off/" for e in report.present)


def test_running_state_skips_done_entries(tmp_path):
    d = _mk_run(tmp_path, state="running")
    report = inspect_run_dir(d)
    # running 状态下 done 级条目不视为缺失
    assert all(e.path != "run_manifest.json" for e in report.missing)
    assert report.ok


def test_inspect_cli_reports_layout_status(tmp_path):
    d = _mk_run(tmp_path, state="failed")

    result = CliRunner().invoke(main, ["inspect", str(d)])

    assert result.exit_code == 0, result.output
    assert "status: OK" in result.output


def test_inspect_cli_fails_for_missing_required_artifacts(tmp_path):
    d = tmp_path / "run"
    d.mkdir()

    result = CliRunner().invoke(main, ["inspect", str(d)])

    assert result.exit_code == 1
    assert "aqa_run.json" in result.output
