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
    return _run(cmd, log_path, timeout, env)


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

    cmd = ["copilot", "-p", prompt, "--allow-all-tools", "--no-color"]
    # Directory sandbox: authorise the workdir + common install prefixes.
    for d in (str(workdir), os.environ.get("HOME", "/root"), "/opt", "/usr", "/tmp"):
        cmd += ["--add-dir", d]
    if not byok and tier.model:
        cmd += ["--model", tier.model]
    _log.info("copilot agent (session=%s, byok=%s, model=%s, timeout=%ss)",
              session_id, byok, tier.model or "default", timeout)
    return _run(cmd, log_path, timeout, env)


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
