#!/usr/bin/env python3
"""Diagnose the hf_dataset upload 403: whoami, token scope, write access, xet.

Reads the SAME env vars the sink resolves (LOCAL_RESULTS_HF_TOKEN / HF_TOKEN),
never prints the token, and reports exactly why a write to the target dataset
succeeds or fails.  Run:  source env.sh && python scripts/diag_hf_token.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = os.environ.get("LOCAL_RESULTS_DATASET", "lvkaokao/lb_local")
TOKEN_ENVS = ("LOCAL_RESULTS_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
              "HUGGINGFACEHUB_API_TOKEN")


def _resolve_token() -> tuple[str | None, str | None]:
    for name in TOKEN_ENVS:
        val = os.environ.get(name)
        if val:
            return val, name
    return None, None


def _mask(tok: str) -> str:
    return f"{tok[:4]}…{tok[-3:]} (len={len(tok)})" if len(tok) > 8 else "<short>"


def main() -> int:
    print(f"== target dataset: {REPO}")
    token, src = _resolve_token()
    if not token:
        print(f"FAIL: no token in any of {TOKEN_ENVS}. `source env.sh` first.")
        return 2
    print(f"== token from ${src} -> {_mask(token)}")

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi(token=token)

    # 1) identity + scope
    try:
        who = api.whoami(token=token)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL whoami: token invalid/expired: {exc}")
        return 3
    name = who.get("name")
    auth = (who.get("auth") or {}).get("accessToken") or {}
    role = auth.get("role") or who.get("auth", {}).get("type")
    print(f"== authenticated as: {name}  (token role/scope: {role})")
    orgs = [o.get("name") for o in who.get("orgs", [])]
    print(f"== orgs: {orgs or '<none>'}")
    owner = REPO.split("/")[0]
    if owner != name and owner not in orgs:
        print(f"WARN: token identity '{name}' is neither the repo owner '{owner}' "
              f"nor a member of it via orgs -> writes will 403 unless granted.")

    # 2) does the dataset exist / is it visible to this token?
    try:
        info = api.dataset_info(REPO, token=token)
        print(f"== dataset exists: id={info.id} private={getattr(info, 'private', '?')}")
    except HfHubHTTPError as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        print(f"WARN dataset_info {code}: {exc}")

    # 3) explicit write-permission probe
    try:
        perm = api.get_repo_permission(REPO, repo_type="dataset", token=token)
        print(f"== repo permission for this token: {perm}")
    except Exception as exc:  # noqa: BLE001 - older hub versions lack this
        print(f"== (get_repo_permission unavailable: {exc})")

    # 4) real minimal write, Xet ON then OFF
    for disable_xet in (False, True):
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        if disable_xet:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        label = "xet-off" if disable_xet else "xet-on"
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "diag.txt"
            f.write_text("aqa hf token write probe\n")
            try:
                api.upload_file(
                    path_or_fileobj=str(f), path_in_repo="_aqa_diag/diag.txt",
                    repo_id=REPO, repo_type="dataset", token=token,
                    commit_message="aqa diag probe",
                )
                print(f"OK  [{label}] upload succeeded -> token HAS write access.")
                try:
                    api.delete_file("_aqa_diag/diag.txt", REPO, repo_type="dataset",
                                    token=token, commit_message="aqa diag cleanup")
                    print(f"    [{label}] cleaned up probe file.")
                except Exception as exc:  # noqa: BLE001
                    print(f"    [{label}] cleanup skipped: {exc}")
                return 0
            except HfHubHTTPError as exc:
                code = getattr(getattr(exc, "response", None), "status_code", None)
                print(f"FAIL [{label}] upload {code}: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL [{label}] upload error: {type(exc).__name__}: {exc}")

    print("\n=> Token reached HF but cannot write. Fix one of:")
    print("   - use a token from an account/org WITH write access to", REPO)
    print("   - ensure the token has the 'write' scope (fine-grained: grant this repo)")
    print("   - if xet-off succeeded above, set `disable_xet: true` on the sink")
    return 1


if __name__ == "__main__":
    sys.exit(main())
