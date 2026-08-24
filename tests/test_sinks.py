"""Tests for result sinks (hf_dataset token/permission handling)."""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from aqa.logging_util import AqaError
from aqa.report.sinks import HfDatasetSink, RunMeta


def _meta():
    return RunMeta(model="org/M", artifact_name="art")


def test_missing_token_raises_clear_error(tmp_path, monkeypatch):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    sink = HfDatasetSink(repo="org/ds")
    with pytest.raises(AqaError, match="no token"):
        sink.upload(tmp_path, _meta())


def test_token_resolution_priority(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from_env")
    monkeypatch.setenv("MY_TOKEN", "from_custom_env")
    # explicit token wins
    assert HfDatasetSink(repo="o/d", token="explicit")._resolve_token() == "explicit"
    # configured token_env beats the standard HF_TOKEN
    assert HfDatasetSink(repo="o/d", token_env="MY_TOKEN")._resolve_token() == "from_custom_env"
    # fallback to HF_TOKEN
    assert HfDatasetSink(repo="o/d")._resolve_token() == "from_env"


class _FakeApi:
    def __init__(self, *, whoami_ok=True, upload_status=None):
        self._whoami_ok = whoami_ok
        self._upload_status = upload_status
        self.uploaded = False

    def whoami(self, token=None):
        if not self._whoami_ok:
            raise RuntimeError("bad token")
        return {"name": "tester"}

    def upload_folder(self, **kwargs):
        if self._upload_status is not None:
            from huggingface_hub.utils import HfHubHTTPError

            class _Resp:
                status_code = self._upload_status
                headers: typing.ClassVar[dict] = {}
                request = None

            err = HfHubHTTPError("denied", response=_Resp())
            raise err
        self.uploaded = True
        # Snapshot the staged tree (temp dir is deleted after upload returns).
        import os

        folder = kwargs.get("folder_path")
        self.last_kwargs = kwargs
        self.staged = {}
        if folder:
            for root, _dirs, files in os.walk(folder):
                for f in files:
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, folder)
                    try:
                        self.staged[rel] = Path(p).read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        self.staged[rel] = None


def _patch_api(monkeypatch, api):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: api)


def test_invalid_token_surfaces_before_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    _patch_api(monkeypatch, _FakeApi(whoami_ok=False))
    with pytest.raises(AqaError, match="invalid or expired"):
        HfDatasetSink(repo="org/ds").upload(tmp_path, _meta())


def test_403_gives_write_permission_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    _patch_api(monkeypatch, _FakeApi(upload_status=403))
    with pytest.raises(AqaError, match="write.*permission|disable_xet"):
        HfDatasetSink(repo="org/ds").upload(tmp_path, _meta())


def test_successful_upload_returns_url(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    api = _FakeApi()
    _patch_api(monkeypatch, api)
    url = HfDatasetSink(repo="org/ds", path_in_repo="results").upload(tmp_path, _meta())
    assert api.uploaded
    assert url == "https://huggingface.co/datasets/org/ds/tree/main/results/art"


def test_disable_xet_sets_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    _patch_api(monkeypatch, _FakeApi())
    HfDatasetSink(repo="org/ds", disable_xet=True).upload(tmp_path, _meta())
    import os

    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"


def _make_run(tmp_path):
    """A realistic aqa run dir: eval + logs/fix records + patches + weights."""
    run = tmp_path / "run"
    (run / "eval").mkdir(parents=True)
    (run / "eval" / "accuracy.json").write_text('{"status": "success"}')
    (run / "logs" / "agent_fixes" / "quantize").mkdir(parents=True)
    (run / "logs" / "agent_fixes" / "quantize" / "attempt_1.log").write_text(
        "token hf_abcdefghijklmnopqrstuvwx leaked here")
    (run / "patches" / "quantize").mkdir(parents=True)
    (run / "patches" / "quantize" / "fix.patch").write_text("--- a\n+++ b\n")
    (run / "quantized").mkdir()
    (run / "quantized" / "model.safetensors").write_bytes(b"\x00" * 8)
    (run / "gate.json").write_text('{"verdict": "pass"}')
    return run


def _full_meta(run):
    return RunMeta(
        model="MyOrg/Base", scheme="W4A16", method="autoround",
        artifact_name="Base-w4a16", run_root=run, run_id="20240101-000000",
        status="Finished", request_filename="Base-w4a16.json",
        extra={"results": {"piqa": 0.78}},
    )


def test_full_run_uploads_logs_patches_excludes_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    api = _FakeApi()
    _patch_api(monkeypatch, api)
    run = _make_run(tmp_path)
    url = HfDatasetSink(repo="org/ds", request_filename=True).upload(run, _full_meta(run))
    staged = api.staged
    # logs + fix records + patches are present
    assert any("attempt_1.log" in k for k in staged)
    assert any("fix.patch" in k for k in staged)
    # weights are excluded
    assert not any("model.safetensors" in k for k in staged)
    # secrets scrubbed
    log = next(v for k, v in staged.items() if "attempt_1.log" in k)
    assert "hf_abcdefghijklmnopqrstuvwx" not in log
    assert "REDACTED" in log
    # aggregate + status write-back present
    assert any(k.endswith("results_20240101-000000.json") for k in staged)
    assert any(k == "status/MyOrg/Base-w4a16.json" or k.endswith("status/MyOrg/Base-w4a16.json")
               for k in staged)
    assert "run_20240101-000000" in url


def test_full_run_disabled_falls_back_to_flat(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    api = _FakeApi()
    _patch_api(monkeypatch, api)
    run = _make_run(tmp_path)
    # full_run=False => flat single-folder push under results/<artifact>
    url = HfDatasetSink(repo="org/ds", full_run=False).upload(run, _full_meta(run))
    assert url == "https://huggingface.co/datasets/org/ds/tree/main/results/Base-w4a16"


def test_full_run_ships_shared_knowledge(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    api = _FakeApi()
    _patch_api(monkeypatch, api)
    run = _make_run(tmp_path)
    # A shared knowledge dir (lessons + patch + stats + a binary to be excluded).
    kn = tmp_path / "knowledge"
    (kn / "lessons").mkdir(parents=True)
    (kn / "lessons" / "quantize.jsonl").write_text('{"signature": "s"}\n')
    (kn / "patches" / "quantize" / "abc").mkdir(parents=True)
    (kn / "patches" / "quantize" / "abc" / "auto_round.patch").write_text("--- a\n+++ b\n")
    (kn / "patches" / "quantize" / "abc" / "meta.json").write_text('{"exc_type": "RuntimeError"}')
    (kn / "stats").mkdir()
    (kn / "stats" / "events.jsonl").write_text('{"outcome": "replay_hit"}\n')
    (kn / "patch_snapshot.bin").write_bytes(b"\x00" * 8)  # excluded (not text artifact)

    meta = _full_meta(run)
    meta.knowledge_dir = kn
    HfDatasetSink(repo="org/ds").upload(run, meta)
    staged = api.staged
    assert any(k == "knowledge/lessons/quantize.jsonl" for k in staged)
    assert any(k.endswith("knowledge/patches/quantize/abc/auto_round.patch") for k in staged)
    assert any(k == "knowledge/stats/events.jsonl" for k in staged)
    assert not any("patch_snapshot.bin" in k for k in staged)  # binary excluded


def test_status_written_reflects_meta(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HF_TOKEN", "t")
    api = _FakeApi()
    _patch_api(monkeypatch, api)
    run = _make_run(tmp_path)
    meta = _full_meta(run)
    meta.status = "Eval Failed"
    HfDatasetSink(repo="org/ds").upload(run, meta)
    status_key = next(k for k in api.staged if k.endswith("status/MyOrg/Base-w4a16.json"))
    assert json.loads(api.staged[status_key])["status"] == "Eval Failed"
