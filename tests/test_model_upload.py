"""Tests for the quantized-model HF upload sink (HfModelSink)."""

from __future__ import annotations

import typing

import pytest

from aqa.logging_util import AqaError
from aqa.report.sinks import HfModelSink, RunMeta, build_model_card


def _meta(**kw):
    base = {"model": "org/Base", "scheme": "W4A16", "method": "autoround",
            "artifact_name": "Base-w4a16"}
    base.update(kw)
    return RunMeta(**base)


class _FakeApi:
    def __init__(self, *, create_status=None, upload_status=None):
        self._create_status = create_status
        self._upload_status = upload_status
        self.created = None
        self.uploaded_kwargs = None

    def _raise(self, status):
        from huggingface_hub.utils import HfHubHTTPError

        class _Resp:
            status_code = status
            headers: typing.ClassVar[dict] = {}
            request = None

        raise HfHubHTTPError("denied", response=_Resp())

    def create_repo(self, **kwargs):
        if self._create_status is not None:
            self._raise(self._create_status)
        self.created = kwargs

    def upload_folder(self, **kwargs):
        if self._upload_status is not None:
            self._raise(self._upload_status)
        self.uploaded_kwargs = kwargs


def _patch_api(monkeypatch, api):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: api)


def _model_dir(tmp_path):
    d = tmp_path / "quantized"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"0")
    return d


def test_missing_repo_raises(tmp_path):
    with pytest.raises(AqaError, match="requires a `repo`"):
        HfModelSink().upload(_model_dir(tmp_path), _meta())


def test_missing_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    with pytest.raises(AqaError, match="model directory not found"):
        HfModelSink(repo="MyOrg").upload(tmp_path / "nope", _meta())


def test_missing_token_raises(tmp_path, monkeypatch):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AqaError, match="no token"):
        HfModelSink(repo="MyOrg").upload(_model_dir(tmp_path), _meta())


def test_successful_upload_returns_url_and_writes_card(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    api = _FakeApi()
    _patch_api(monkeypatch, api)
    md = _model_dir(tmp_path)
    url = HfModelSink(repo="MyOrg").upload(md, _meta())
    assert url == "https://huggingface.co/MyOrg/Base-w4a16"
    assert api.created["repo_id"] == "MyOrg/Base-w4a16"
    assert api.created["repo_type"] == "model"
    assert api.created["exist_ok"] is True
    # ignore patterns are applied so secrets/logs never leak
    assert "accuracy.json" in api.uploaded_kwargs["ignore_patterns"]
    assert (md / "README.md").exists()


def test_repo_name_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    api = _FakeApi()
    _patch_api(monkeypatch, api)
    HfModelSink(repo="MyOrg", repo_name="custom-name").upload(_model_dir(tmp_path), _meta())
    assert api.created["repo_id"] == "MyOrg/custom-name"


def test_403_gives_write_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    _patch_api(monkeypatch, _FakeApi(create_status=403))
    with pytest.raises(AqaError, match="denied .403.|write"):
        HfModelSink(repo="MyOrg").upload(_model_dir(tmp_path), _meta())


def test_model_card_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    _patch_api(monkeypatch, _FakeApi())
    md = _model_dir(tmp_path)
    HfModelSink(repo="MyOrg", model_card=False).upload(md, _meta())
    assert not (md / "README.md").exists()


def test_build_model_card_contents():
    card = build_model_card(
        _meta(), repo_name="Base-w4a16",
        results={"piqa": 0.78, "hellaswag": 0.55},
    )
    assert "base_model:" in card
    assert "org/Base" in card
    assert "W4A16" in card
    assert "AutoRound" in card  # method link resolved
    assert "| piqa | 0.7800 |" in card
    assert "hellaswag" in card
