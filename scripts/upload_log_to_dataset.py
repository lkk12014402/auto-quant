#!/usr/bin/env python3
"""Upload a run log to the HF dataset path recorded *inside that log*.

A run log (e.g. ``qwen3.5-35b-a3b.log``) contains the line the sink prints when
it finishes uploading results::

    ... aqa.report.sinks: results uploaded to
    https://huggingface.co/datasets/<repo>/tree/main/<path> (status=Finished)

This standalone helper parses that line, derives ``<repo>`` + ``<path>`` and
uploads the log file itself into the same folder, so the log lives next to the
results it produced.  It touches nothing in ``aqa/`` and adds no dependency
beyond ``huggingface_hub`` (already required by the project).

Usage:
    source env.sh
    python scripts/upload_log_to_dataset.py qwen3.5-35b-a3b.log

Options:
    --dest-name NAME   filename to store in the dataset (default: log basename)
    --url URL          override: upload to this dataset tree URL instead of
                       parsing the log
    --prefer-any       upload to the last "results uploaded" line even if it is
                       not marked ``status=Finished``
    --dry-run          parse + print the resolved target, upload nothing
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

# The sink log line + its embedded dataset tree URL.
_UPLOAD_RE = re.compile(
    r"results uploaded to\s+"
    r"(?P<url>https://huggingface\.co/datasets/\S+?)"
    r"(?:\s+\(status=(?P<status>[^)]*)\))?\s*$"
)
# .../datasets/<repo_id>/tree/<revision>/<path_in_repo>
_URL_RE = re.compile(
    r"https://huggingface\.co/datasets/"
    r"(?P<repo>[^/]+/[^/]+)/tree/(?P<rev>[^/]+)/(?P<path>.+?)/?$"
)
_TOKEN_ENVS = ("LOCAL_RESULTS_HF_TOKEN", "HF_TOKEN",
               "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def _resolve_token() -> str | None:
    for name in _TOKEN_ENVS:
        val = os.environ.get(name)
        if val:
            return val
    return None


def _connect_hint(exc: Exception) -> str:
    """Turn a low-level TLS/connection failure into an actionable message.

    A common cause here is a proxy mismatch: ``huggingface_hub`` (via ``httpx``)
    prefers the upper-case ``HTTPS_PROXY``/``HTTP_PROXY``, while tools like curl
    may use the lower-case ``https_proxy``/``http_proxy``.  If the upper-case one
    points at a proxy that drops TLS, the upload fails with an ``SSL_EOF``.
    """
    msg = str(exc)
    lines = [f"upload failed: {type(exc).__name__}: {msg}"]
    up = os.environ.get("HTTPS_PROXY") or ""
    lo = os.environ.get("https_proxy") or ""
    is_conn = "SSL" in msg or "EOF" in msg or "ConnectError" in type(exc).__name__
    if is_conn and up and lo and up != lo:
        lines.append(
            f"hint: HTTPS_PROXY ({up}) differs from https_proxy ({lo}); the "
            "upper-case one (used by huggingface_hub) may be failing TLS. Try:\n"
            "      export HTTPS_PROXY=\"$https_proxy\" HTTP_PROXY=\"$http_proxy\"\n"
            "      then re-run this script."
        )
    elif is_conn:
        lines.append(
            "hint: this looks like a network/proxy/TLS problem, not a script bug. "
            "Verify HF is reachable through your proxy, then re-run."
        )
    return "\n".join(lines)


def _find_upload_url(log_text: str, *, prefer_finished: bool = True) -> str:
    """Return the dataset tree URL from the log's 'results uploaded' lines.

    Prefers the last ``status=Finished`` line (the completed main run); falls
    back to the last upload line of any status when ``prefer_finished`` is False
    or nothing is marked Finished.
    """
    finished: list[str] = []
    any_upload: list[str] = []
    for line in log_text.splitlines():
        m = _UPLOAD_RE.search(line.strip())
        if not m:
            continue
        url = m.group("url")
        any_upload.append(url)
        if (m.group("status") or "").strip().lower() == "finished":
            finished.append(url)
    if prefer_finished and finished:
        return finished[-1]
    if any_upload:
        return any_upload[-1]
    raise SystemExit(
        "no 'results uploaded to https://huggingface.co/datasets/...' line found "
        "in the log; pass --url <dataset tree URL> to upload explicitly."
    )


def _parse_tree_url(url: str) -> tuple[str, str, str]:
    """Split a dataset tree URL into (repo_id, revision, path_in_repo)."""
    m = _URL_RE.match(url.strip())
    if not m:
        raise SystemExit(f"cannot parse dataset tree URL: {url}")
    return m.group("repo"), unquote(m.group("rev")), unquote(m.group("path"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="path to the run .log file to upload")
    ap.add_argument("--dest-name", default=None,
                    help="filename to store in the dataset (default: log basename)")
    ap.add_argument("--url", default=None,
                    help="dataset tree URL to upload into (skip log parsing)")
    ap.add_argument("--prefer-any", action="store_true",
                    help="use the last upload line even if not status=Finished")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + print the target, upload nothing")
    args = ap.parse_args(argv)

    if not args.log.is_file():
        raise SystemExit(f"log file not found: {args.log}")

    url = args.url or _find_upload_url(
        args.log.read_text(errors="replace"), prefer_finished=not args.prefer_any)
    repo_id, revision, path_in_repo = _parse_tree_url(url)
    dest_name = args.dest_name or args.log.name
    path_in_repo = f"{path_in_repo}/{dest_name}"

    print(f"log        : {args.log}")
    print(f"repo_id    : {repo_id}  (dataset)")
    print(f"revision   : {revision}")
    print(f"dest path  : {path_in_repo}")
    target_url = f"https://huggingface.co/datasets/{repo_id}/blob/{revision}/{path_in_repo}"
    print(f"target     : {target_url}")

    if args.dry_run:
        print("dry-run: nothing uploaded.")
        return 0

    token = _resolve_token()
    if not token:
        raise SystemExit(
            "no HF token in env (looked for: " + ", ".join(_TOKEN_ENVS) + "). "
            "Run `source env.sh` first.")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("huggingface_hub not installed; `pip install huggingface_hub`.")

    api = HfApi(token=token)
    try:
        who = api.whoami().get("name", "?")
        print(f"authenticated as {who}; uploading ...")
    except Exception as exc:  # noqa: BLE001 - preflight only; don't block the actual upload
        print(f"warning: whoami preflight failed ({type(exc).__name__}); continuing to upload ...")
    try:
        commit = api.upload_file(
            path_or_fileobj=str(args.log),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            commit_message=f"add run log {dest_name}",
        )
    except Exception as exc:  # re-raised as SystemExit with an actionable hint
        raise SystemExit(_connect_hint(exc)) from exc
    print(f"uploaded -> {getattr(commit, 'commit_url', target_url)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
