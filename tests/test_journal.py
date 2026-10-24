"""Tests for the external-agent run journal (aqa/session/journal.py + CLI)."""

from __future__ import annotations

import json

from click.testing import CliRunner

from aqa import cli
from aqa.session import journal_entries
from aqa.session.journal import (
    JOURNAL_FILE,
    append_journal,
    mirror_decision,
    render_markdown,
)


def test_append_bootstraps_and_persists(tmp_path):
    run = tmp_path / "run"
    entry = append_journal(run, title="Plan A", body="do X", kind="plan", phase="PLAN")
    assert entry["id"] == "j001"
    # event stream created + journal.md rendered
    events = journal_entries(run)
    assert len(events) == 1 and events[0]["title"] == "Plan A"
    md = (run / JOURNAL_FILE).read_text()
    assert "Plan A" in md and "do X" in md and "PLAN" in md


def test_append_only_revision_preserves_history(tmp_path):
    run = tmp_path / "run"
    append_journal(run, title="Scheme mixed", kind="plan", phase="PLAN")
    rev = append_journal(run, title="Switch to global", kind="revise",
                         phase="QUANT", supersedes="j001")
    assert rev["id"] == "j002" and rev["supersedes"] == "j001"
    entries = journal_entries(run)
    # both the original and the revision survive (append-only)
    assert [e["id"] for e in entries] == ["j001", "j002"]
    md = render_markdown(run)
    assert "Scheme mixed" in md and "Switch to global" in md
    assert "supersedes: j001" in md


def test_markdown_rebuilds_deterministically(tmp_path):
    run = tmp_path / "run"
    append_journal(run, title="E1", kind="note")
    append_journal(run, title="E2", kind="note")
    assert render_markdown(run) == render_markdown(run)


def test_bad_kind_and_empty_title_rejected(tmp_path):
    run = tmp_path / "run"
    try:
        append_journal(run, title="x", kind="bogus")
        assert False, "expected ValueError on bad kind"
    except ValueError:
        pass
    try:
        append_journal(run, title="   ", kind="note")
        assert False, "expected ValueError on empty title"
    except ValueError:
        pass


def test_mirror_decision_requires_state(tmp_path):
    run = tmp_path / "run"
    append_journal(run, title="D", kind="decision", phase="QUANT")
    # no state.json yet -> no-op, returns False
    assert mirror_decision(run, phase="QUANT", what="D", why="why") is False
    # create a minimal state.json -> mirror appends a decision
    sdir = run / "session"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "state.json").write_text(json.dumps({"decisions": []}))
    assert mirror_decision(run, phase="QUANT", what="D", why="picked 5") is True
    data = json.loads((sdir / "state.json").read_text())
    assert data["decisions"][-1]["what"] == "D"


def test_cli_journal_add_show(tmp_path):
    run = tmp_path / "run"
    r = CliRunner()
    res = r.invoke(cli.main, ["journal", "add", "-r", str(run), "-k", "plan",
                              "-p", "PLAN", "-t", "Title", "-b", "Body line"])
    assert res.exit_code == 0, res.output
    assert "j001" in res.output
    res = r.invoke(cli.main, ["journal", "show", "-r", str(run)])
    assert res.exit_code == 0 and "Title" in res.output and "Body line" in res.output


def test_cli_body_from_file(tmp_path):
    run = tmp_path / "run"
    bodyfile = tmp_path / "plan.md"
    bodyfile.write_text("multi\nline\nplan")
    r = CliRunner()
    res = r.invoke(cli.main, ["journal", "add", "-r", str(run), "-t", "FromFile",
                              "-b", f"@{bodyfile}"])
    assert res.exit_code == 0, res.output
    assert "multi\nline\nplan" in render_markdown(run)
