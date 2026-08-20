"""SshExecutor: run the full pipeline on a remote GPU host over SSH.

Ships *this* package to the host, builds a docker image (optionally from the
repo's own ``docker/gpu.dockerfile`` on an NGC base), and runs ``aqa run`` inside
the container with ``executor.type=local``.  Supports (optionally single-hop)
password or key/agent auth via paramiko.

Credentials are read ONLY from environment variables named by ``auth.password_env``
(never stored in config) -- an open-source hard constraint.
"""

from __future__ import annotations

import io
import os
import re
import shlex
import tarfile
from pathlib import Path

from ..config import RunConfig, SshConfig
from ..logging_util import AqaError, get_logger
from . import ProcessState, RunResult
from ._remote import (
    build_container_bootstrap,
    build_inner_config,
    build_remote_script,
    collect_secret_env,
    make_run_id,
    repo_root,
)

_log = get_logger("executor.ssh")

# Directories excluded from the shipped repo tarball (keep it small + no secrets).
_TAR_EXCLUDE = {
    ".git",
    ".venv",
    "venv",
    "results",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    ".idea",
    ".vscode",
}


def build_ssh_argv(ssh: SshConfig) -> list[str]:
    """Build a printable ssh argv (for dry-run display) tunneling through hops."""
    base = ["ssh", "-o", "StrictHostKeyChecking=no"]
    jump_targets = [hop.host for hop in ssh.jump]
    if jump_targets:
        base += ["-J", ",".join(jump_targets)]
    if ssh.auth.method == "key" and ssh.auth.key_file:
        base += ["-i", os.path.expanduser(ssh.auth.key_file)]
    argv = list(base)
    if ssh.auth.method == "password":
        pw_env = ssh.auth.password_env
        if not pw_env or pw_env not in os.environ:
            raise AqaError(
                f"ssh password auth requires env var {pw_env!r} to be set (never inline secrets)"
            )
        argv = ["sshpass", "-e", *argv]
    argv.append(ssh.host)
    return argv


def _tar_repo() -> bytes:
    """gzip-tar the installed repo (source + recipes + docker) for staging."""
    root = repo_root()
    buf = io.BytesIO()

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if set(Path(info.name).parts) & _TAR_EXCLUDE:
            return None
        return info

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in sorted(root.iterdir()):
            if item.name in _TAR_EXCLUDE:
                continue
            tar.add(str(item), arcname=item.name, filter=_filter)
    return buf.getvalue()


def _dockerfile_bytes(run_cfg: RunConfig) -> bytes | None:
    """Resolve the Dockerfile to ship (``local`` => repo default, path => custom).

    ``local`` selects the accelerator-appropriate shipped Dockerfile: the NGC-based
    ``gpu.dockerfile`` for CUDA, or the Intel-base ``xpu.dockerfile`` for XPU.
    """
    df = (run_cfg.executor.image.dockerfile or "").strip()
    if not df:
        return None
    if df in ("local", "default"):
        accelerator = run_cfg.executor.ssh.accelerator if run_cfg.executor.ssh else "cuda"
        name = "xpu.dockerfile" if accelerator == "xpu" else "gpu.dockerfile"
        return (repo_root() / "docker" / name).read_bytes()
    path = Path(df).expanduser()
    if not path.exists():
        raise AqaError(f"executor.image.dockerfile path not found: {path}")
    return path.read_bytes()


# ─────────────────────────── paramiko transport ────────────────────────────
def _split_host(host: str) -> tuple[str, str, int]:
    user = None
    if "@" in host:
        user, host = host.split("@", 1)
    port = 22
    if ":" in host:
        host, port_s = host.rsplit(":", 1)
        port = int(port_s)
    return user or os.environ.get("USER", "root"), host, port


def _auth_kwargs(auth) -> dict:
    kw: dict = {"allow_agent": auth.method == "agent", "look_for_keys": auth.method == "agent"}
    if auth.method == "password":
        if not auth.password_env or auth.password_env not in os.environ:
            raise AqaError(f"ssh password auth requires env var {auth.password_env!r} to be set")
        kw["password"] = os.environ[auth.password_env]
    elif auth.method == "key" and auth.key_file:
        kw["key_filename"] = os.path.expanduser(auth.key_file)
    return kw


def _connect(paramiko, *, host, port, user, auth, timeout, sock=None):
    """Connect a paramiko client, falling back to keyboard-interactive for
    password auth against servers that only offer that method (PAM)."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, port=port, username=user, timeout=timeout,
                       auth_timeout=timeout, banner_timeout=timeout, sock=sock,
                       **_auth_kwargs(auth))
        return client
    except paramiko.AuthenticationException:
        if auth.method != "password":
            raise
        # Retry over the raw transport with keyboard-interactive (same password).
        password = os.environ[auth.password_env]
        transport = client.get_transport()
        if transport is None or not transport.is_active():
            client.close()
            raise
        try:
            transport.auth_interactive_dumb(user, lambda *a, **k: [password])
        except Exception as exc:
            client.close()
            raise AqaError(
                f"ssh authentication failed for {user}@{host}: check the login user "
                f"and the value of ${auth.password_env}"
            ) from exc
        return client


def _open_client(ssh: SshConfig, timeout: int = 30):
    """Open a paramiko SSHClient to ``ssh.host``, optionally via one jump hop."""
    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover - optional dep
        raise AqaError(
            "remote ssh dispatch needs paramiko: pip install 'autoquant-agent[ssh]'"
        ) from exc

    sock = None
    if ssh.jump:
        hop = ssh.jump[0]
        h_user, h_host, h_port = _split_host(hop.host)
        jump = _connect(paramiko, host=h_host, port=h_port, user=h_user,
                        auth=hop.auth, timeout=timeout)
        _t_user, t_host, t_port = _split_host(ssh.host)
        sock = jump.get_transport().open_channel("direct-tcpip", (t_host, t_port), ("", 0))
        if len(ssh.jump) > 1:
            _log.warning("only the first jump hop is wired here; use aqa-executor-intel "
                         "for deeper chains")

    user, host, port = _split_host(ssh.host)
    return _connect(paramiko, host=host, port=port, user=user, auth=ssh.auth,
                    timeout=timeout, sock=sock)


def _sftp_makedirs(sftp, path: str) -> None:
    cur = ""
    for part in path.strip("/").split("/"):
        cur = f"{cur}/{part}" if cur else (f"/{part}" if path.startswith("/") else part)
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def _sftp_fetch_tree(sftp, remote_dir: str, local_dir: Path) -> int:
    """Recursively download ``remote_dir`` into ``local_dir``; return file count.

    Silently returns 0 if the remote dir does not exist (e.g. run failed before
    producing any results).
    """
    import stat as _stat

    try:
        entries = sftp.listdir_attr(remote_dir)
    except OSError:
        return 0
    count = 0
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        rpath = f"{remote_dir}/{entry.filename}"
        lpath = local_dir / entry.filename
        if _stat.S_ISDIR(entry.st_mode):
            count += _sftp_fetch_tree(sftp, rpath, lpath)
        else:
            try:
                sftp.get(rpath, str(lpath))
                count += 1
            except OSError:
                _log.warning("could not fetch %s", rpath)
    return count


def _exec_stream(client, command: str, *, poll: float = 0.5) -> int:
    """Run ``command`` remotely, streaming combined stdout/stderr live; return rc.

    lm_eval / vllm report progress with tqdm bars that use carriage returns
    (``\\r``) instead of newlines, and phases like model load or a long eval run
    for minutes emitting no ``\\n`` at all.  Splitting only on ``\\n`` therefore
    makes those phases look hung (while the GPU is clearly busy).  We instead:

      * treat both ``\\r`` and ``\\n`` as line delimiters, so each progress update
        surfaces as it arrives, and
      * block on ``recv`` with a short timeout (no busy-poll) and, when the stream
        goes briefly idle, flush any pending partial line so a paused progress bar
        or newline-less prompt still shows up.
    """
    chan = client.get_transport().open_session()
    chan.get_pty()
    chan.exec_command(command)
    chan.settimeout(poll)
    buf = b""
    while True:
        try:
            data = chan.recv(65536)
        except TimeoutError:
            data = b""
        if data:
            buf += data
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()  # trailing partial line; completes on a later read
            for line in parts:
                if line:
                    _log.info("[remote] %s", line.decode(errors="replace"))
        else:
            if buf:
                _log.info("[remote] %s", buf.decode(errors="replace"))
                buf = b""
            if chan.exit_status_ready():
                break
    if buf:
        _log.info("[remote] %s", buf.decode(errors="replace"))
    return chan.recv_exit_status()


class SshExecutor:
    name = "ssh"

    def __init__(self, **options):
        self.options = options
        self.remote_root = options.get("remote_root", "~/aqa-runs")

    def run(self, run_cfg: RunConfig, *, dry_run: bool = False, resume: bool = False) -> RunResult:
        ssh = run_cfg.executor.ssh
        if ssh is None:
            raise AqaError("executor.type=ssh requires an `ssh:` config block")

        run_id = make_run_id(run_cfg)
        inner_cfg = build_inner_config(run_cfg, run_id)
        secrets = collect_secret_env(run_cfg)
        script = build_remote_script(
            run_cfg, run_id, has_secrets=bool(secrets), remote_root=self.remote_root
        )
        dockerfile = _dockerfile_bytes(run_cfg)

        printable = " ".join(shlex.quote(a) for a in build_ssh_argv(ssh))
        out_dir = Path(run_cfg.output.dir)

        if dry_run:
            _log.info("[CMD] %s 'bash %s/%s/_dispatch.sh'", printable, self.remote_root, run_id)
            _log.info("[dry-run] run_id=%s secrets=%s dockerfile=%s",
                      run_id, sorted(secrets), "local/custom" if dockerfile else "none")
            _log.info("[dry-run] --- remote dispatch script ---\n%s", script)
            _log.info("[dry-run] --- inner run.yaml ---\n%s", inner_cfg)
            return RunResult(output_dir=out_dir, state=ProcessState.DONE)

        _log.info("[CMD] %s  (staging + docker build/run, run_id=%s)", printable, run_id)
        client = _open_client(ssh, timeout=int(self.options.get("timeout", 30)))
        try:
            sftp = client.open_sftp()
            root = self._expand(client, self.remote_root)
            work = f"{root}/{run_id}"
            _sftp_makedirs(sftp, work)

            staged: dict[str, tuple[bytes, int]] = {
                "_repo.tgz": (_tar_repo(), 0o600),
                "_run.yaml": (inner_cfg.encode(), 0o600),
                "_bootstrap.sh": (build_container_bootstrap(run_cfg).encode(), 0o755),
                "_dispatch.sh": (script.encode(), 0o700),
            }
            if dockerfile is not None:
                staged["_gpu.dockerfile"] = (dockerfile, 0o600)
            if secrets:
                env_txt = "".join(f"{k}={v}\n" for k, v in secrets.items())
                staged["_secrets.env"] = (env_txt.encode(), 0o600)

            for name, (content, mode) in staged.items():
                path = f"{work}/{name}"
                with sftp.open(path, "wb") as fh:
                    fh.write(content)
                sftp.chmod(path, mode)

            rc = _exec_stream(client, f"bash {shlex.quote(work)}/_dispatch.sh")

            # Pull the run's results/logs back so the user can inspect eval logs
            # (and gate/accuracy json) locally without SSHing into the box.
            remote_results = f"{work}/repo/results/{run_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            fetched = _sftp_fetch_tree(sftp, remote_results, out_dir)
            if fetched:
                _log.info("fetched %d result file(s) into %s", fetched, out_dir)
            elif rc != 0:
                _log.warning("no result files found on remote at %s", remote_results)
            sftp.close()
        finally:
            client.close()

        state = ProcessState.DONE if rc == 0 else ProcessState.FAILED
        if rc != 0:
            log_hint = next(iter(out_dir.rglob("*.eval.log")), None)
            _log.error("remote dispatch failed (rc=%s); remote run %s. %s",
                       rc, run_id,
                       f"see {log_hint}" if log_hint else f"inspect {out_dir}")
        return RunResult(output_dir=out_dir, state=state)

    @staticmethod
    def _expand(client, path: str) -> str:
        if not path.startswith("~"):
            return path
        _in, out, _err = client.exec_command(f"eval echo {shlex.quote(path)}")
        return out.read().decode().strip() or path

    def status(self, output_dir: Path) -> ProcessState:
        return ProcessState.UNKNOWN

    def stop(self, output_dir: Path) -> bool:
        return False

    @staticmethod
    def detect(output_dir: Path) -> bool:
        return False
