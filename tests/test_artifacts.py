"""Tests for aqa.report.artifacts (AGENTS.md 硬流程 artifact contract)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aqa.report.artifacts import check_artifacts, format_report

_ANALYSIS = """# X 结构分析 + 量化方案

## 0. 查阅清单（必填，带日期/版本）
## 1. 架构判定
## 4. 量化方案
## 5. 评估计划
"""

_REPORT_GO = """# X 回溯报告

## 1. 结果总览

- gate 结论：**GO**

## 5. 经验沉淀路由表

| 经验类型 | 落点 | 机制 | 本次内容 | 状态 |
|---|---|---|---|---|
| 报错→修法 | knowledge | 随 run 回传 | 坑① | 已落地：knowledge/x |
| 通用坑 | AGENTS.md §9 | 改文档 | 无 | 本次无 |

## 6. 后续行动
"""

_REPORT_UNFILLED = _REPORT_GO.replace("已落地：knowledge/x", "").replace("本次无", "")


def _fill_run_dir(run_dir: Path, *, report: str = _REPORT_GO,
                  run_yaml: str = "model: org/X\nquant: {scheme: MXFP4}\n") -> None:
    (run_dir / "analysis.md").write_text(_ANALYSIS)
    (run_dir / "setup_venv.sh").write_text("#!/usr/bin/env bash\n")
    (run_dir / "requirements.txt").write_text("torch==0.0.0\n")
    (run_dir / "journal.md").write_text("# journal\n")
    (run_dir / "report.md").write_text(report)
    (run_dir / "run.yaml").write_text(run_yaml)
    (run_dir / "gate.json").write_text('{"verdict": "GO", "reasons": []}')


def _errors(report) -> list[str]:
    return [f.message for f in report.findings if f.level == "error"]


def _warns(report) -> list[str]:
    return [f.message for f in report.findings if f.level == "warn"]


def test_complete_run_dir_passes(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    report = check_artifacts(tmp_path)
    assert report.ok, format_report(report)
    assert not report.findings


def test_missing_file_is_error(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "analysis.md").unlink()
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any("analysis.md 缺失" in m for m in _errors(report))


def test_missing_section_is_error(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "analysis.md").write_text(_ANALYSIS.replace("## 0. 查阅清单（必填，带日期/版本）\n", ""))
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any("查阅清单" in m for m in _errors(report))


def test_run_yaml_required(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "run.yaml").unlink()
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any("run.yaml 缺失" in m for m in _errors(report))


def test_gate_verdict_mismatch(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path, report=_REPORT_GO.replace("**GO**", "**NO-GO**"))
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any("gate" in m and "GO" in m for m in _errors(report))


def test_gate_verdict_absent_in_report(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path, report=_REPORT_GO.replace("**GO**", "**待定**"))
    report = check_artifacts(tmp_path)
    assert not report.ok


def test_routing_table_unfilled_is_error(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path, report=_REPORT_UNFILLED)
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any("路由表未填写" in m for m in _errors(report))


def test_routing_table_partially_filled_is_warn(tmp_path: Path) -> None:
    report_text = _REPORT_GO.replace("本次无", "", 1)
    _fill_run_dir(tmp_path, report=report_text)
    report = check_artifacts(tmp_path)
    assert report.ok
    assert any("未填状态" in m for m in _warns(report))


def test_model_card_required_only_when_upload(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    assert check_artifacts(tmp_path).ok  # no upload -> not required

    (tmp_path / "run.yaml").write_text(
        "model: org/X\nprovider:\n  upload: {repo: org/X-MXFP4-AutoRound, token_env: HF_TOKEN}\n")
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any("model_card.md 缺失" in m for m in _errors(report))

    (tmp_path / "model_card.md").write_text(
        "# card\n\nSigned-off-by: Jane Smith <jane.smith@intel.com>\n")
    assert check_artifacts(tmp_path).ok


@pytest.mark.parametrize("card, needle", [
    ("# card\n", "末行须为"),
    ("Signed-off-by: Jane Smith <jane.smith@intel.com>\n\n# card\n", "末行须为"),
    ("# card\n\nSigned-off-by: Jane Smith\n", "末行须为"),
    ("# card\n\nSigned-off-by: {Name} <{email}>\n", "末行须为"),
    ("# card\n\nSigned-off-by: Full Name <full.name@example.com>\n", "占位身份"),
])
def test_model_card_requires_signoff_last_line(tmp_path: Path, card: str, needle: str) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "model_card.md").write_text(card)
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any(needle in m for m in _errors(report))


def test_no_gate_json_skips_verdict_check(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "gate.json").unlink()
    assert check_artifacts(tmp_path).ok


def test_format_report_smoke(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "journal.md").unlink()
    out = format_report(check_artifacts(tmp_path))
    assert "INCOMPLETE" in out and "journal.md" in out


@pytest.mark.parametrize("name", ["run.yaml", "run.yml", "x_run.yaml"])
def test_any_top_level_yaml_counts(tmp_path: Path, name: str) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "run.yaml").rename(tmp_path / name)
    assert check_artifacts(tmp_path).ok
