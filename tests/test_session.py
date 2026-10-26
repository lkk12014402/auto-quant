"""Tests for the session layer (state machine, manifest, events) + pipeline resume.

Covers docs/rfc/0001-reproducible-session-state-machine.md P1.
"""

from __future__ import annotations

import json

import pytest

from aqa.session import (
    Phase,
    SessionState,
    Status,
    append_event,
    load_session_manifest,
    open_session,
    read_events,
)
from aqa.session.state import state_path


# ── state machine ─────────────────────────────────────────────────────────────
def test_state_new_marks_all_pending_and_round_trips(tmp_path):
    plan = [Phase.QUANT, Phase.EVAL, Phase.CLOSE]
    st = SessionState.new("run-1", plan)
    assert st.phase == "QUANT"
    assert all(v == Status.PENDING.value for v in st.phases.values())

    st.enter(Phase.QUANT)
    assert st.status_of(Phase.QUANT) == Status.RUNNING.value
    st.done(Phase.QUANT)
    assert st.is_done(Phase.QUANT)
    st.commit(tmp_path)

    reloaded = SessionState.from_dict(json.loads(state_path(tmp_path).read_text()))
    assert reloaded.is_done(Phase.QUANT)
    assert reloaded.status_of(Phase.EVAL) == Status.PENDING.value


def test_commit_is_atomic_no_tmp_left_behind(tmp_path):
    st = SessionState.new("run-1", [Phase.EVAL])
    st.commit(tmp_path)
    assert state_path(tmp_path).exists()
    assert not (tmp_path / "state.json.tmp").exists()


def test_events_append_and_read(tmp_path):
    append_event(tmp_path, "phase_enter", phase="QUANT")
    append_event(tmp_path, "phase_done", phase="QUANT")
    evs = read_events(tmp_path)
    assert [e["type"] for e in evs] == ["phase_enter", "phase_done"]
    assert all("ts" in e for e in evs)


# ── session facade / resume precondition ──────────────────────────────────────
def _cfg(tmp_path, **over):
    from aqa.config import RunConfig

    base = {
        "model": "org/m",
        "provider": {"type": "quantize", "quant": {"scheme": "MXFP4"}},
        "eval": {"benchmarks": ["piqa"], "backend": "vllm", "baseline": False},
        "output": {"dir": str(tmp_path), "sinks": []},
    }
    base.update(over)
    return RunConfig.model_validate(base)


def test_open_session_fresh_writes_manifest_and_state(tmp_path):
    cfg = _cfg(tmp_path)
    plan = [Phase.QUANT, Phase.EVAL, Phase.CLOSE]
    sess = open_session(tmp_path, cfg, "run-1", plan)
    manifest = load_session_manifest(tmp_path / "session")
    assert manifest["run_id"] == "run-1"
    assert manifest["model"] == "org/m"
    assert "env_lock" in manifest and "run_config" in manifest
    assert state_path(sess.dir).exists()


def test_open_session_resume_without_manifest_refuses(tmp_path):
    from aqa.logging_util import AqaError

    cfg = _cfg(tmp_path)
    with pytest.raises(AqaError, match="no session manifest"):
        open_session(tmp_path, cfg, "run-1", [Phase.EVAL], resume=True)


def test_should_skip_requires_done_and_artifact(tmp_path):
    cfg = _cfg(tmp_path)
    sess = open_session(tmp_path, cfg, "run-1", [Phase.QUANT, Phase.EVAL])
    # pending phase never skips, even with artifact present
    assert not sess.should_skip(Phase.QUANT, artifact_present=True)
    sess.state.done(Phase.QUANT)
    assert sess.should_skip(Phase.QUANT, artifact_present=True)
    assert not sess.should_skip(Phase.QUANT, artifact_present=False)  # half-baked -> rerun


# ── pipeline resume integration ───────────────────────────────────────────────
def test_pipeline_resume_skips_quantize(tmp_path, monkeypatch):
    """A resumed run with a checkpoint present must not re-run quantization."""
    import aqa.pipeline as P
    from aqa.providers.base import ResolvedModel

    monkeypatch.setattr(P, "_dispatch_sinks", lambda *a, **k: None)
    monkeypatch.setattr(P, "_make_knowledge", lambda rc: None)
    monkeypatch.setattr(P, "_finalize_knowledge", lambda k: None)
    monkeypatch.setattr(P, "_maybe_upload_model", lambda *a, **k: None)

    def _result(scores):
        from aqa.eval.registry import BenchmarkResult, EvalResult
        r = EvalResult(model="org/m", backend="vllm")
        for name, sc in scores.items():
            r.add(BenchmarkResult(name=name, score=sc, metric="acc"))
        return r

    monkeypatch.setattr(P, "_run_evaluate", lambda *a, **k: _result({"piqa": 0.8}))

    # first (fresh) run: quantize is invoked, produces a checkpoint
    quant_calls: list[str] = []

    def _resolve_fresh(run_cfg, out_dir, num_gpus, knowledge=None, run_id="", heal=None,
                       hot_start="", session=None):
        quant_calls.append("quantized")
        export = out_dir / "quantized"
        export.mkdir(parents=True, exist_ok=True)
        (export / "config.json").write_text("{}")
        return ResolvedModel(model=str(export), kind="path")

    monkeypatch.setattr(P, "_resolve_model", _resolve_fresh)

    cfg = _cfg(tmp_path)
    assert P.run_pipeline(cfg, tmp_path) == P.ProcessState.DONE
    assert quant_calls == ["quantized"]

    from aqa.session.state import SessionState, state_path
    st = SessionState.from_dict(json.loads(state_path(tmp_path / "session").read_text()))
    assert st.is_done(Phase.QUANT) and st.is_done(Phase.EVAL)

    # second (resume) run: _resolve_model must NOT be called again
    def _resolve_boom(*a, **k):
        raise AssertionError("quantize must be skipped on resume")

    monkeypatch.setattr(P, "_resolve_model", _resolve_boom)
    assert P.run_pipeline(cfg, tmp_path, resume=True) == P.ProcessState.DONE

    st2 = SessionState.from_dict(json.loads(state_path(tmp_path / "session").read_text()))
    assert st2.status_of(Phase.QUANT) == Status.SKIPPED.value
    evs = read_events(tmp_path / "session")
    assert any(e["type"] == "phase_skip" and e.get("phase") == "QUANT" for e in evs)

    # third (resume) run: SKIPPED must still count as done. Regression: skip()
    # demoted DONE -> SKIPPED while should_skip() only accepted DONE, so a second
    # resume silently re-ran quantization.
    assert P.run_pipeline(cfg, tmp_path, resume=True) == P.ProcessState.DONE
    assert quant_calls == ["quantized"]

    # Resume must not mint a new run_id: state/manifest/env_lock/manifest all
    # have to agree, otherwise the run directory carries several identities.
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["run_id"] == st.run_id
    assert json.loads((tmp_path / "env_lock.json").read_text())["run_id"] == st.run_id
    assert json.loads((tmp_path / "quant_plan.json").read_text())["run_id"] == st.run_id
