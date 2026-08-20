"""Tests for run-artifact staging + secret scrubbing (scrub.py)."""

from __future__ import annotations

from aqa.report.scrub import sanitize_text, stage_run_artifacts


def test_sanitize_redacts_tokens():
    assert "REDACTED" in sanitize_text("key hf_abcdefghijklmnopqrstuvwx here")
    assert "hf_abcdefghijklmnopqrstuvwx" not in sanitize_text("hf_abcdefghijklmnopqrstuvwx")
    out = sanitize_text("Authorization: Bearer supersecretvalue")
    assert "supersecretvalue" not in out
    assert "REDACTED" in out


def test_stage_excludes_weights_and_caches(tmp_path):
    src = tmp_path / "run"
    (src / "logs").mkdir(parents=True)
    (src / "logs" / "a.log").write_text("ok")
    (src / "quantized").mkdir()
    (src / "quantized" / "model.safetensors").write_bytes(b"\x00")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (src / "weights.bin").write_bytes(b"\x00")
    dst = tmp_path / "staged"
    copied = stage_run_artifacts(src, dst)
    assert "logs/a.log" in copied
    assert not (dst / "quantized").exists()
    assert not (dst / "__pycache__").exists()
    assert not (dst / "weights.bin").exists()


def test_stage_sanitizes_text_files(tmp_path):
    src = tmp_path / "run"
    src.mkdir()
    (src / "attempt.log").write_text("leak hf_abcdefghijklmnopqrstuvwx end")
    dst = tmp_path / "staged"
    stage_run_artifacts(src, dst)
    assert "hf_abcdefghijklmnopqrstuvwx" not in (dst / "attempt.log").read_text()


def test_stage_extra_excluded_dirs(tmp_path):
    src = tmp_path / "run"
    (src / "Base-w4a16").mkdir(parents=True)
    (src / "Base-w4a16" / "dup.json").write_text("{}")
    (src / "keep.json").write_text("{}")
    dst = tmp_path / "staged"
    copied = stage_run_artifacts(src, dst, extra_excluded_dirs={"Base-w4a16"})
    assert "keep.json" in copied
    assert not (dst / "Base-w4a16").exists()
