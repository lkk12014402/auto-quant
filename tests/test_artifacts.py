"""Tests for aqa.report.artifacts (AGENTS.md 硬流程 artifact contract)."""

from __future__ import annotations

import json
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
def test_model_card_draft_is_warning_without_upload(
        tmp_path: Path, card: str, needle: str) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "model_card.md").write_text(card)
    report = check_artifacts(tmp_path)
    assert report.ok
    assert any(needle in m for m in _warns(report))


@pytest.mark.parametrize("card, needle", [
    ("# card\n", "末行须为"),
    ("# card\n\nSigned-off-by: {Name} <{email}>\n", "末行须为"),
    ("# card\n\nSigned-off-by: Full Name <full.name@example.com>\n", "占位身份"),
])
def test_model_card_bad_signoff_is_error_when_upload(
        tmp_path: Path, card: str, needle: str) -> None:
    _fill_run_dir(
        tmp_path,
        run_yaml="model: org/X\nprovider:\n  upload: {repo: org/X}\n")
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


def test_recipe_manifests_are_not_treated_as_run_config(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "qwen3-8-27b.yaml").write_text(
        "schema_version: '1.0'\nmodel: {id: qwen3-8-27b, source: x}\n"
        "environments: {}\noperations: {}\nvariants: []\nresult_tables: []\n")
    report = check_artifacts(tmp_path)
    assert report.ok
    assert "qwen3-8-27b.yaml" not in report.present


def test_upload_detected_with_recipe_sidecar(tmp_path: Path) -> None:
    _fill_run_dir(
        tmp_path,
        run_yaml="model: org/X\nprovider:\n  upload: {repo: org/X}\n")
    (tmp_path / "qwen3-8-27b.yaml").write_text(
        "schema_version: '1.0'\nmodel: {id: qwen3-8-27b, source: x}\n"
        "environments: {}\noperations: {}\nvariants: []\nresult_tables: []\n")
    report = check_artifacts(tmp_path)
    assert any("model_card.md 缺失" in m for m in _errors(report))


def test_report_placeholders_are_error(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path, report=_REPORT_GO + "\n<!-- RESULTS_TABLE -->\n")
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any("模板占位符" in m for m in _errors(report))


def test_run_id_mismatch_is_warn(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "env_lock.json").write_text('{"run_id": "run-a"}')
    (tmp_path / "run_manifest.json").write_text('{"run_id": "run-b"}')
    report = check_artifacts(tmp_path)
    assert report.ok
    assert any("run_id" in m for m in _warns(report))


def test_recipe_kb_failed_is_error(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "recipe_kb.json").write_text(json.dumps({
        "record": {"status": "failed", "error": "Read-only file system"}}))
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any("Auto KB" in m for m in _errors(report))


def test_card_fixed_section_drift_is_warn(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "model_card.md").write_text(
        "# card\n\n## Ethical Considerations and Limitations\n\n"
        "Custom text.\n\nSigned-off-by: Jane Smith <jane.smith@intel.com>\n")
    report = check_artifacts(tmp_path)
    assert report.ok
    assert any("Ethical" in m for m in _warns(report))


def test_smoke_eval_card_is_warn(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path, run_yaml="model: org/X\neval: {limit: 8}\n")
    (tmp_path / "model_card.md").write_text(
        "# card\n\nSigned-off-by: Jane Smith <jane.smith@intel.com>\n")
    report = check_artifacts(tmp_path)
    assert report.ok
    assert any("smoke" in m for m in _warns(report))


def test_knowledge_sync_failed_is_error(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "knowledge_sync.json").write_text(json.dumps({
        "status": "failed", "errors": ["auth denied"]}))
    report = check_artifacts(tmp_path)
    assert not report.ok
    assert any("knowledge_sync" in m for m in _errors(report))


def test_knowledge_sync_local_only_is_warn(tmp_path: Path) -> None:
    _fill_run_dir(tmp_path)
    (tmp_path / "knowledge_sync.json").write_text(json.dumps({
        "status": "local_only", "errors": []}))
    report = check_artifacts(tmp_path)
    assert report.ok
    assert any("knowledge 仅本地" in m for m in _warns(report))
