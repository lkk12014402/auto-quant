"""Stage a run directory for upload: exclude model weights / caches and scrub
secrets from text artifacts.

Ported from ``local_dispatch/upload_results_hf_dataset.py`` so the aqa results
sink writes back the *same* shape local_dispatch did (logs, agent fix records,
patches, summaries), minus the quantized weights, with secrets redacted.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

# Suffixes we sanitize as UTF-8 text; everything else is copied byte-for-byte.
TEXT_SUFFIXES = {
    ".json", ".jsonl", ".log", ".md", ".txt", ".sh", ".py",
    ".yaml", ".yml", ".patch", ".diff", ".csv", ".cfg", ".ini",
}

# Directories that must never be uploaded (weights + caches + vcs).
EXCLUDED_DIRS = {
    "quantized", "quantized_model", "__pycache__", ".git", ".cache",
    ".pytest_cache", ".venv", "venv",
}

# Weight/blob files that may live outside a weights dir.
EXCLUDED_SUFFIXES = {
    ".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".h5", ".ckpt",
}

SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_\-]{15,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+"),
    re.compile(r"(?i)(x-access-token:)[^@\s/]+"),
    # generic credentials in URL userinfo: scheme://user:secret@host
    re.compile(r"(https?://[^/\s:@]+:)[^@\s/]+(@)"),
)


def sanitize_text(text: str) -> str:
    """Redact tokens/secrets from a text artifact before it leaves the machine."""
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            def _sub(m: re.Match, n: int = pattern.groups) -> str:
                # keep every capture group, inserting the redaction after group 1
                return m.group(1) + "***REDACTED***" + "".join(
                    m.group(i) or "" for i in range(2, n + 1))
            text = pattern.sub(_sub, text)
        else:
            text = pattern.sub("***REDACTED***", text)
    return text


def stage_run_artifacts(
    src: Path,
    dst: Path,
    *,
    extra_excluded_dirs: set[str] | None = None,
    sanitize: bool = True,
) -> list[str]:
    """Copy ``src`` into ``dst`` excluding weights/caches and scrubbing text.

    Returns the list of copied paths (relative to ``src``).
    """
    src = Path(src)
    dst = Path(dst)
    excluded = set(EXCLUDED_DIRS)
    if extra_excluded_dirs:
        excluded |= set(extra_excluded_dirs)
    copied: list[str] = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in excluded]
        root_path = Path(root)
        rel_root = root_path.relative_to(src)
        for name in files:
            source = root_path / name
            if source.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            rel = rel_root / name
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if sanitize and source.suffix.lower() in TEXT_SUFFIXES:
                try:
                    text = source.read_text(encoding="utf-8", errors="replace")
                    target.write_text(sanitize_text(text), encoding="utf-8")
                except OSError:
                    continue
            else:
                try:
                    shutil.copy2(source, target)
                except OSError:
                    continue
            copied.append(str(rel))
    return copied
