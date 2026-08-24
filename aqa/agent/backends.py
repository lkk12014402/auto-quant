"""Invoke the agent CLI (OpenClaw / Copilot) to attempt a fix.

Ports ``run_openclaw_fix`` / ``run_copilot_fix`` from
``local_dispatch/agent/agent_backends.sh``.  Each backend runs a headless,
tool-enabled agent against the working tree with a timeout; the combined output
is written to ``log_path`` for :mod:`aqa.agent.verdict` to parse.

Auth is env-var only: OpenClaw reads ``MINIMAX_API_KEY``; Copilot uses BYOK
(reusing ``MINIMAX_API_KEY``) or ``COPILOT_GITHUB_TOKEN``.  Nothing is inlined.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from ..logging_util import get_logger
from .tiers import Tier

_log = get_logger("agent.backend")

DEFAULT_TIMEOUT = 1200


class AgentUnavailable(RuntimeError):
    """Raised when the tier's agent CLI is not installed / not authenticated."""


def _run(cmd: list[str], log_path: Path, timeout: int, env: dict[str, str]) -> int:
    """Run the agent CLI, teeing its output live to the console AND ``log_path``.

    The agent takes minutes; streaming (like local_dispatch's ``tee``) lets a
    remote ``aqa run`` see the agent's reasoning/tool calls in real time instead
    of a silent gap between "attempt N" and the verdict.  Returns the exit code
    (``124`` on timeout, after the agent's process tree is killed).
    """
    from ..proc import TIMEOUT_RC, stream_subprocess

    rc = stream_subprocess(cmd, log_path, env=env, timeout=timeout)
    if rc == TIMEOUT_RC:
        _log.warning("agent timed out after %ss", timeout)
    return rc


def _archive_agent_session(
    backend: str, session_id: str, log_path: Path, env: dict[str, str],
) -> None:
    """Copy the agent CLI's own session transcript next to ``log_path``.

    The streamed stdout in ``log_path`` is the rich-UI view and is often truncated
    when a slow run is SIGTERM'd on timeout. The CLI also persists a structured
    session (copilot ``events.jsonl`` / openclaw ``<id>.jsonl``) with the full
    reasoning + tool calls. We archive it under ``logs/agent_fixes/<phase>/`` so it
    is uploaded with the results and fetched back -- mirroring local_dispatch, whose
    report reads these to recover the agent's diagnosis/fix record.
    """
    import shutil

    stem = log_path.stem
    try:
        if backend == "copilot":
            state_root = env.get("COPILOT_SESSION_STATE_DIR") or str(
                Path(env.get("HOME", "/root")) / ".copilot" / "session-state"
            )
            events = Path(state_root) / session_id / "events.jsonl"
            if events.is_file():
                shutil.copy2(events, log_path.with_name(f"{stem}.events.jsonl"))
        else:  # openclaw
            sessions_dir = env.get("OPENCLAW_SESSIONS_DIR") or str(
                Path(env.get("HOME", "/root")) / ".openclaw" / "agents" / "main" / "sessions"
            )
            session = Path(sessions_dir) / f"{session_id}.jsonl"
            if session.is_file():
                shutil.copy2(session, log_path.with_name(f"{stem}.session.jsonl"))
    except OSError as exc:  # best-effort archival; never fail the fix on this
        _log.warning("could not archive %s session %s: %s", backend, session_id, exc)


def _openclaw_fix(prompt: str, log_path: Path, session_id: str, tier: Tier,
                  timeout: int, workdir: Path) -> int:
    if shutil.which("openclaw") is None:
        raise AgentUnavailable("openclaw CLI not found")
    env = {**os.environ, **tier.env}
    cmd = [
        "openclaw", "agent", "--local",
        "--session-id", session_id,
        "--message", prompt,
        "--timeout", str(timeout),
    ]
    _log.info("openclaw agent (session=%s, timeout=%ss)", session_id, timeout)
    rc = _run(cmd, log_path, timeout, env)
    _archive_agent_session("openclaw", session_id, log_path, env)
    return rc


def _copilot_session_id(label: str) -> str:
    """Copilot's CLI requires ``--session-id`` to be a valid UUID.

    Fix-loop session ids are human-readable labels (e.g. ``fix_evaluate_7369e9fa``)
    that the CLI rejects ("not a valid UUID, so a new session cannot be created").
    If the label is already a UUID, pass it through; otherwise derive a stable
    UUIDv5 from it so the same label always maps to the same session id (and the
    session-state we archive afterwards is found under that same uuid dir).
    """
    try:
        return str(uuid.UUID(label))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aqa-agent-fix/{label}"))


def _copilot_fix(prompt: str, log_path: Path, session_id: str, tier: Tier,
                 timeout: int, workdir: Path) -> int:
    if shutil.which("copilot") is None:
        raise AgentUnavailable("copilot CLI not found")
    env = {**os.environ, **tier.env}
    byok = tier.byok or bool(env.get("COPILOT_PROVIDER_BASE_URL") or env.get("COPILOT_MINIMAX"))
    if byok:
        env.setdefault("COPILOT_PROVIDER_API_KEY", env.get("MINIMAX_API_KEY", ""))
        if not env["COPILOT_PROVIDER_API_KEY"]:
            raise AgentUnavailable("copilot BYOK needs MINIMAX_API_KEY / COPILOT_PROVIDER_API_KEY")
        if tier.model:
            env.setdefault("COPILOT_MODEL", tier.model)
    elif not (env.get("COPILOT_GITHUB_TOKEN") or env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")):
        raise AgentUnavailable("copilot needs COPILOT_GITHUB_TOKEN (or BYOK provider)")

    # Copilot requires a UUID session id; coerce the human-readable label.
    cli_session_id = _copilot_session_id(session_id)
    # A human-readable transcript (--share) + the CLI's own log dir land next to the
    # attempt log so they upload with the results; --session-id ties the run to the
    # session-state we archive afterwards.
    share_md = log_path.with_name(f"{log_path.stem}.share.md")
    copilot_logs = log_path.with_name(f"{log_path.stem}_copilot_logs")
    cmd = [
        "copilot", "-p", prompt, "--allow-all-tools", "--no-color",
        "--session-id", cli_session_id,
        "--share", str(share_md),
        "--log-dir", str(copilot_logs),
        "--log-level", "error",
    ]
    # Directory sandbox: authorise the workdir + common install prefixes. The attempt-log
    # dir is added explicitly so the agent can read prior attempts' transcripts/prompts
    # (the prior-state block points it at those files) even if it lives outside workdir.
    for d in (str(workdir), str(log_path.parent), os.environ.get("HOME", "/root"),
              "/opt", "/usr", "/tmp"):
        cmd += ["--add-dir", d]
    if not byok and tier.model:
        cmd += ["--model", tier.model]
    _log.info("copilot agent (session=%s, byok=%s, model=%s, timeout=%ss)",
              cli_session_id, byok, tier.model or "default", timeout)
    rc = _run(cmd, log_path, timeout, env)
    _archive_agent_session("copilot", cli_session_id, log_path, env)
    return rc


def run_agent_fix(tier: Tier, prompt: str, log_path: Path, *,
                  session_id: str | None = None, workdir: Path | None = None,
                  timeout: int = DEFAULT_TIMEOUT) -> int:
    """Dispatch a fix attempt to the tier's backend; return the process rc.

    Raises :class:`AgentUnavailable` if the backend CLI is missing or unauthenticated
    (the caller treats that as "skip this tier / escalate").
    """
    session_id = session_id or str(uuid.uuid4())
    workdir = Path(workdir or Path.cwd())
    if tier.backend == "openclaw":
        return _openclaw_fix(prompt, log_path, session_id, tier, timeout, workdir)
    return _copilot_fix(prompt, log_path, session_id, tier, timeout, workdir)
