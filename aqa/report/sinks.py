"""Result sinks: pluggable destinations for run artifacts.

Every sink implements :class:`ResultSink`.  ``local_fs`` is the built-in default;
``hf_dataset`` ports the historic HF-dataset upload (with secret scrubbing).  New
destinations (s3, wandb, ...) are new classes + entry-point registration.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..logging_util import AqaError, get_logger

_log = get_logger("report.sinks")

# Env vars huggingface_hub honours, in resolution order, when no explicit token
# is configured on the sink.
_HF_TOKEN_ENVS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


import re


def _safe_name(value: str) -> str:
    """Filesystem/repo-safe slug (mirrors local_dispatch's ``safe_name``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value or "").strip("-") or "unknown"


def resolve_hf_token(token: str | None, token_env: str | None) -> str | None:
    """Explicit ``token`` > configured ``token_env`` > standard HF env vars."""
    if token:
        return token
    candidates = [token_env] if token_env else []
    candidates += list(_HF_TOKEN_ENVS)
    for name in candidates:
        val = name and os.environ.get(name)
        if val:
            return val
    return None


@dataclass
class RunMeta:
    model: str
    scheme: str | None = None
    method: str | None = None
    artifact_name: str = ""
    request_filename: str = ""
    # Whole-run context (set by the pipeline) so the dataset sink can write back
    # the full run — logs, agent fix records, patches — the way local_dispatch did.
    run_root: Path | None = None
    run_id: str = ""
    status: str = ""
    # Explicit org override for dataset/model upload paths. When unset, derived
    # from `model` (org/name ids); local model paths fall back to "unknown".
    org_override: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Shared fix-knowledge dir (lessons + replayable patches + replay stats). When
    # set on a full-run upload it rides the results dataset under ``knowledge/`` so
    # a later run can sync it back — cross-machine sharing without a separate repo.
    knowledge_dir: Path | None = None

    @property
    def org(self) -> str:
        """Owner slug derived from ``org/name`` model ids (else ``unknown``).

        An explicit ``org`` override wins; otherwise derive from ``model``.
        Local model paths (``/models/...`` or ``./models/...``) bucket under
        ``local``; bare ids without an org bucket under ``unknown``.
        """
        if self.org_override:
            return self.org_override
        if "/" not in self.model:
            return "unknown"
        first = self.model.split("/", 1)[0]
        return first if first not in ("", ".") else "local"


@runtime_checkable
class ResultSink(Protocol):
    name: str

    def upload(self, run_dir: Path, meta: RunMeta) -> str:
        ...


class LocalFsSink:
    name = "local_fs"

    def __init__(self, path: str = "./results", **_: Any):
        self.path = Path(path)

    def upload(self, run_dir: Path, meta: RunMeta) -> str:
        dest = self.path / (meta.artifact_name or Path(run_dir).name)
        resolved_dest = dest.resolve()
        dest.mkdir(parents=True, exist_ok=True)
        if Path(run_dir).resolve() != dest.resolve():
            def ignore_destination(path: str, names: list[str]) -> list[str]:
                try:
                    component = resolved_dest.relative_to(Path(path).resolve()).parts[0]
                except (ValueError, IndexError):
                    return []
                return [name for name in names if name == component]

            for item in Path(run_dir).iterdir():
                if item.resolve() == resolved_dest:
                    continue
                target = dest / item.name
                if item.is_dir():
                    shutil.copytree(
                        item, target, dirs_exist_ok=True, ignore=ignore_destination)
                else:
                    shutil.copy2(item, target)
        (dest / "run_meta.json").write_text(json.dumps(meta.__dict__, indent=2, default=str))
        _log.info("results written to %s", dest)
        return str(dest)


class HfDatasetSink:
    name = "hf_dataset"

    def __init__(
        self,
        repo: str | None = None,
        path_in_repo: str = "results",
        *,
        token: str | None = None,
        token_env: str | None = None,
        endpoint: str | None = None,
        disable_xet: bool = False,
        full_run: bool = True,
        sanitize: bool = True,
        status: bool = True,
        request_filename: bool = False,
        **_: Any,
    ):
        self.repo = repo
        self.path_in_repo = path_in_repo
        self.token = token
        self.token_env = token_env
        self.endpoint = endpoint
        self.disable_xet = disable_xet
        # Leaderboard-parity behaviour (matches local_dispatch's uploader):
        self.full_run = full_run          # upload the whole run (logs/patches/fixes)
        self.sanitize = sanitize          # scrub secrets from text artifacts
        self.status = status              # write status/<org>/<request_filename>.json
        self.request_filename = request_filename  # embed source request filename

    def _resolve_token(self) -> str | None:
        """Explicit token > configured ``token_env`` > standard HF env vars."""
        return resolve_hf_token(self.token, self.token_env)

    def upload(self, run_dir: Path, meta: RunMeta) -> str:
        if not self.repo:
            raise AqaError("hf_dataset sink requires a `repo` (e.g. org/dataset)")
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - optional dep
            raise AqaError("hf_dataset sink needs `huggingface_hub`") from exc

        token = self._resolve_token()
        if not token:
            raise AqaError(
                "hf_dataset upload has no token. Set HF_TOKEN in the environment "
                "(it is forwarded into the remote container) or configure "
                "`token_env: <VAR>` on the sink. The token must have *write* "
                f"access to dataset '{self.repo}'."
            )
        # Some write tokens lack Xet access; fall back to plain LFS on request.
        if self.disable_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        api = HfApi(token=token, endpoint=self.endpoint)
        # Fail fast with an actionable message instead of a raw 403 from upload.
        self._check_write_access(api, token)

        # Whole-run write-back (logs / agent fix records / patches / summaries),
        # matching local_dispatch's uploader, when the pipeline supplies run_root.
        if self.full_run and meta.run_root is not None:
            return self._upload_full_run(api, token, meta)
        return self._upload_flat(api, token, run_dir, meta)

    def _upload_flat(self, api, token: str, run_dir: Path, meta: RunMeta) -> str:
        target = f"{self.path_in_repo}/{meta.artifact_name or Path(run_dir).name}"
        self._upload_folder(api, token, Path(run_dir), target)
        url = f"https://huggingface.co/datasets/{self.repo}/tree/main/{target}"
        _log.info("results uploaded to %s", url)
        return url

    def _upload_full_run(self, api, token: str, meta: RunMeta) -> str:
        """Stage + upload the whole run (weights excluded, secrets scrubbed) plus
        an aggregate ``results_<id>.json`` and a ``status/`` write-back."""
        import json
        import tempfile
        import time

        from .scrub import stage_run_artifacts

        org = _safe_name(meta.org)
        artifact = _safe_name(meta.artifact_name or Path(meta.run_root).name)
        run_id = meta.run_id or time.strftime("%Y%m%d-%H%M%S")
        request_filename = meta.request_filename or f"{artifact}.json"
        if not request_filename.endswith(".json"):
            request_filename += ".json"

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            result_root = staging / self.path_in_repo / org / artifact
            staged_run = result_root / f"run_{run_id}"
            copied = stage_run_artifacts(
                Path(meta.run_root), staged_run, sanitize=self.sanitize,
                extra_excluded_dirs={meta.artifact_name, artifact} if meta.artifact_name else None)

            aggregate = {
                "status": meta.status or "Finished",
                "pipeline": "aqa",
                "model": meta.model,
                "scheme": meta.scheme,
                "method": meta.method,
                "artifact": artifact,
                "run_id": run_id,
                "results": meta.extra.get("results", {}),
                "num_files": len(copied),
            }
            if self.request_filename:
                aggregate["request_filename"] = request_filename
            result_root.mkdir(parents=True, exist_ok=True)
            (result_root / f"results_{run_id}.json").write_text(
                json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")

            if self.status:
                status_data = {
                    "status": meta.status or "Finished",
                    "model": meta.model,
                    "artifact": artifact,
                    "request_filename": request_filename,
                    "run_id": run_id,
                }
                status_path = staging / "status" / org / request_filename
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(
                    json.dumps(status_data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")

            # Shared fix-knowledge rides the results dataset under ``knowledge/``.
            if meta.knowledge_dir and Path(meta.knowledge_dir).is_dir():
                self._stage_knowledge(Path(meta.knowledge_dir), staging / "knowledge")

            self._upload_folder(api, token, staging, path_in_repo=None,
                                commit_message=f"Add run for {artifact} ({run_id})")

        target = f"{self.path_in_repo}/{org}/{artifact}/run_{run_id}"
        url = f"https://huggingface.co/datasets/{self.repo}/tree/main/{target}"
        _log.info("results uploaded to %s (status=%s)", url, meta.status or "Finished")
        return url

    @staticmethod
    def _stage_knowledge(src: Path, dst: Path) -> None:
        """Copy the knowledge tree (lessons/patches/stats) into the upload staging.

        Only lightweight text artifacts (``.jsonl``/``.patch``/``.json``) are shipped;
        the snapshot cache and any binaries are excluded.
        """
        import shutil

        keep = (".jsonl", ".patch", ".json")
        for p in src.rglob("*"):
            if not p.is_file() or p.suffix not in keep:
                continue
            rel = p.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)

    def _upload_folder(self, api, token: str, folder: Path, path_in_repo,
                       commit_message: str | None = None) -> None:
        from huggingface_hub.utils import HfHubHTTPError
        try:
            api.upload_folder(
                repo_id=self.repo,
                repo_type="dataset",
                folder_path=str(folder),
                path_in_repo=path_in_repo,
                token=token,
                commit_message=commit_message,
            )
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 403:
                raise AqaError(
                    f"hf_dataset upload to '{self.repo}' was denied (403). The token "
                    "reached HF but lacks *write* permission on this dataset. Check "
                    "that the token belongs to an account/org with write access and "
                    "has the 'write' scope. If this is a Xet 403 on a valid write "
                    "token, retry with `disable_xet: true` on the sink."
                ) from exc
            raise AqaError(f"hf_dataset upload to '{self.repo}' failed: {exc}") from exc

    def _check_write_access(self, api, token: str) -> None:
        """Pre-flight the token so misconfiguration surfaces clearly, not as a 403."""
        try:
            who = api.whoami(token=token)
        except Exception as exc:  # invalid/expired token
            raise AqaError(
                f"hf_dataset token is invalid or expired (whoami failed: {exc}). "
                "Provide a valid HF token with write access."
            ) from exc
        name = who.get("name") if isinstance(who, dict) else None
        _log.info("hf_dataset authenticated as %s -> %s", name, self.repo)


# Files that must never be pushed into a public model repo (secrets, logs,
# eval/session artifacts).  Ported from local_dispatch/auto_quant/upload_model_hf.py.
_MODEL_IGNORE_PATTERNS = [
    "*.pyc", "__pycache__/**", "venv/**", ".venv/**", "*.egg-info", ".git/**",
    "session_*.jsonl", "session_*.md", "quant_summary.json", "summary.json",
    "accuracy.json", "request.json", "logs/**", "lm_eval_results/**", "*.log",
    "*_prompt.txt", "hf_account_usage.json", "quantize_script.py",
]

_METHOD_LINKS = {
    "autoround": ("AutoRound", "https://github.com/intel/auto-round"),
    "model_free": ("AutoRound", "https://github.com/intel/auto-round"),
    "rtn": ("AutoRound", "https://github.com/intel/auto-round"),
    "gptq": ("GPTQ", "https://github.com/AutoGPTQ/AutoGPTQ"),
    "awq": ("AWQ", "https://github.com/mit-han-lab/llm-awq"),
}


def build_model_card(meta: RunMeta, *, repo_name: str,
                     results: dict[str, float] | None = None,
                     commands: dict[str, Any] | None = None,
                     versions: dict[str, str] | None = None) -> str:
    """Render a compact README.md model card for a quantized model repo."""
    base = meta.model or "Unknown"
    scheme = meta.scheme or "Unknown"
    method_key = (meta.method or "").lower()
    method_display, method_url = _METHOD_LINKS.get(method_key, (meta.method or "Unknown", None))
    gen_by = f"[{method_display}]({method_url})" if method_url else method_display
    # HF validates the `base_model` front-matter entry as an `org/name` repo id. When the
    # source is a local checkpoint path (the normal case when quantizing from disk) the
    # whole upload is rejected with "Invalid metadata in README.md", so omit the field and
    # render the path as plain text rather than a broken `https://huggingface.co//models/..`
    # link. Callers should pass `output.report_model` to get a real base_model back.
    is_repo_id = bool(re.fullmatch(r"[^/: ]+/[^/ ]+", base))
    base_md = f"[{base}](https://huggingface.co/{base})" if is_repo_id else f"`{base}`"
    tags = ["quantized", scheme.lower(), method_display.lower(), "autoquant-agent"]
    lines = [
        "---",
        *(["base_model:", f"- {base}"] if is_repo_id else []),
        "pipeline_tag: text-generation",
        "tags:",
        *[f"- {t}" for t in tags],
        "---",
        "",
        f"# {repo_name}",
        "",
        "## Model Details",
        "",
        (f"This is a **{scheme}** quantization of "
         f"{base_md}, generated by {gen_by}. "
         "Please follow the license of the original model."),
        "",
        "## Quantization Details",
        "",
        f"- **Scheme:** {scheme}",
        f"- **Method:** {method_display}",
        f"- **Base model:** {base_md}",
    ]
    if results:
        lines += ["", "## Evaluation", "", "| Benchmark | Score |", "|---|---|"]
        lines += [f"| {name} | {score:.4f} |" for name, score in sorted(results.items())]
    # Reproduce & Environment（与上游 HEAD 对齐）：卡片必须自带**实际执行过的**量化/评估
    # 命令与关键包版本，否则拿到权重也无法复现。命令与版本由调用点从该 run 的
    # run_manifest.json 读出（`commands` / `environment.packages`），不在此重新拼串。
    if commands or versions:
        lines += ["", "## Reproduce & Environment", ""]
        qcmd = (commands or {}).get("quantize")
        if qcmd:
            lines += ["Quantized with:", "", "```bash", qcmd, "```", ""]
        ecmds = (commands or {}).get("evaluate") or []
        if ecmds:
            lines += ["Evaluated with:", "", "```bash", *ecmds, "```", ""]
        if versions:
            keep = [k for k in ("auto-round", "auto_round", "lm-eval", "lm_eval",
                                "compressed-tensors", "transformers", "torch", "vllm")
                    if k in versions]
            seen: set[str] = set()
            rows = []
            for k in keep:
                canon = k.replace("_", "-").lower()
                if canon in seen:
                    continue
                seen.add(canon)
                rows.append(f"| {k} | {versions[k]} |")
            if rows:
                lines += ["| Package | Version |", "|---|---|", *rows, ""]
        lines += ["", ("Inference: load this repo with a vllm build that supports the "
                       f"{scheme} format (day-0 architectures may need the model card's "
                       "custom image/branch — check the quantize command above for the "
                       "environment used).")]
    lines += ["", ("_Produced with [autoquant-agent](https://github.com/) — "
              "agent-driven quantize + evaluate + self-heal._"), ""]
    return "\n".join(lines)


class HfModelSink:
    """Upload a **quantized model** to a Hugging Face *model* repo.

    Minimal single-token/single-org port of local_dispatch's ``upload_model_hf.py``:
    ``create_repo`` + ``upload_folder`` (with secret/log ignore patterns) plus a
    generated model card.  Multi-account failover + quota ledger are intentionally
    left out of this first version.

    Unlike the result sinks, ``upload``'s first arg is the **model directory**
    (the quantized checkpoint), not the eval run dir; the pipeline invokes it with
    the quantize provider's output.
    """

    name = "hf_model"

    def __init__(
        self,
        repo: str | None = None,          # org / owner the repo lives under
        *,
        repo_name: str | None = None,     # repo id under `repo`; default = artifact name
        token: str | None = None,
        token_env: str | None = None,
        private: bool = False,
        endpoint: str | None = None,
        disable_xet: bool = False,
        model_card: bool = True,
        **_: Any,
    ):
        self.repo = repo
        self.repo_name = repo_name
        self.token = token
        self.token_env = token_env
        self.private = private
        self.endpoint = endpoint
        self.disable_xet = disable_xet
        self.model_card = model_card

    def _resolve_token(self) -> str | None:
        return resolve_hf_token(self.token, self.token_env)

    def upload(self, model_dir: Path, meta: RunMeta) -> str:
        if not self.repo:
            raise AqaError("hf_model sink requires a `repo` (the org/owner to upload under)")
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise AqaError(f"hf_model: model directory not found: {model_dir}")
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import HfHubHTTPError
        except ImportError as exc:  # pragma: no cover - optional dep
            raise AqaError("hf_model sink needs `huggingface_hub`") from exc

        token = self._resolve_token()
        if not token:
            raise AqaError(
                "hf_model upload has no token. Set HF_TOKEN (forwarded into the "
                "remote container) or configure `token_env: <VAR>` on the sink. The "
                f"token must have *write* access under '{self.repo}'."
            )
        if self.disable_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        repo_name = self.repo_name or meta.artifact_name or model_dir.name
        repo_id = f"{self.repo}/{repo_name}"

        if self.model_card:
            self._write_model_card(model_dir, meta, repo_name)

        api = HfApi(token=token, endpoint=self.endpoint)
        try:
            api.create_repo(repo_id=repo_id, token=token, repo_type="model",
                            exist_ok=True, private=self.private)
            api.upload_folder(
                repo_id=repo_id, repo_type="model", folder_path=str(model_dir),
                token=token, ignore_patterns=_MODEL_IGNORE_PATTERNS,
                commit_message=f"Upload quantized model {repo_name}",
            )
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 403:
                raise AqaError(
                    f"hf_model upload to '{repo_id}' was denied (403). The token "
                    "reached HF but lacks *write* permission under this org/owner. "
                    "Use a token whose account can write to "
                    f"'{self.repo}' (create/write scope)."
                ) from exc
            raise AqaError(f"hf_model upload to '{repo_id}' failed: {exc}") from exc
        url = f"https://huggingface.co/{repo_id}"
        _log.info("quantized model uploaded to %s", url)
        return url

    def _card_commands_versions(self, meta: RunMeta) -> tuple[dict | None, dict | None]:
        """卡片要的 `commands` / `versions`：优先 meta.extra，否则回落到该 run 的
        run_manifest.json —— pipeline 的调用点只往 extra 里塞 results，
        没有这两项就会退化成"只有分数、无法复现"的卡片（实测 7 个仓库都这样）。
        manifest 里 `commands`（quantize/evaluate）与 `environment.packages`
        本来就是全的，直接读即可，不重新拼串。"""
        extra = meta.extra if isinstance(meta.extra, dict) else {}
        commands = extra.get("commands")
        versions = extra.get("versions")
        if (not commands or not versions) and getattr(meta, "run_root", None):
            m: dict = {}
            try:
                m = json.loads((Path(meta.run_root) / "run_manifest.json").read_text())
            except (OSError, ValueError):
                m = {}
            commands = commands or m.get("commands")
            versions = versions or (m.get("environment") or {}).get("packages")
        return commands, versions

    def _write_model_card(self, model_dir: Path, meta: RunMeta, repo_name: str) -> None:
        # Prefer the run's reviewed model_card.md (hard-flow step ④, signed by
        # the operator).  Only fall back to the compact generated card for
        # direct API callers that have no run directory.
        run_root = getattr(meta, "run_root", None)
        run_card = (Path(run_root) / "model_card.md") if run_root else None
        if run_card is not None and run_card.is_file():
            from .model_card import signoff_problem

            problem = signoff_problem(run_card.read_text(errors="replace"))
            if problem:
                raise AqaError(problem)
            try:
                shutil.copy2(run_card, model_dir / "README.md")
                return
            except OSError as exc:  # never block the upload on a card write
                _log.warning("hf_model: could not copy model_card.md: %s", exc)
        results = meta.extra.get("results") if isinstance(meta.extra, dict) else None
        commands, versions = self._card_commands_versions(meta)
        try:
            (model_dir / "README.md").write_text(
                build_model_card(meta, repo_name=repo_name, results=results,
                                 commands=commands, versions=versions), encoding="utf-8")
        except OSError as exc:  # never block the upload on a card write
            _log.warning("hf_model: could not write model card: %s", exc)
