"""Serve a (quantized) model as an OpenAI-compatible endpoint via ``vllm serve``.

E0 base for agentic/terminal evaluation: those benchmarks need the model behind a
live ``/v1/chat/completions`` endpoint (so a solver can hold multi-turn
conversations), not the lm_eval subprocess path.  :class:`VllmEndpoint` launches
``vllm serve``, waits for ``/health``, and tears the server down on exit.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging_util import AqaError, get_logger

if TYPE_CHECKING:
    from typing import Self

_log = get_logger("agent_eval.endpoint")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VllmEndpoint:
    """Launch and manage a ``vllm serve`` OpenAI-compatible server as a context manager."""

    def __init__(
        self,
        model: str,
        *,
        served_name: str | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        num_gpus: int = 1,
        extra_args: list[str] | None = None,
        startup_timeout: float = 900.0,
        log_path: Path | None = None,
    ):
        self.model = model
        self.served_name = served_name or model
        self.host = host
        self.port = port or _free_port()
        self.num_gpus = num_gpus
        self.extra_args = list(extra_args or [])
        self.startup_timeout = startup_timeout
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    @staticmethod
    def available() -> bool:
        return shutil.which("vllm") is not None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def build_argv(self) -> list[str]:
        argv = [
            "vllm", "serve", self.model,
            "--host", self.host, "--port", str(self.port),
            "--served-model-name", self.served_name,
        ]
        if self.num_gpus > 1:
            argv += ["--tensor-parallel-size", str(self.num_gpus)]
        argv += self.extra_args
        return argv

    def _health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self._health_url(), timeout=2) as resp:
                return 200 <= resp.status < 300
        except Exception:  # noqa: BLE001 - not up yet
            return False

    def start(self) -> Self:
        if not self.available():
            raise AqaError("vllm is not installed; `pip install vllm`")
        argv = self.build_argv()
        _log.info("[CMD] %s", " ".join(argv))
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("w")
            out = self._log_fh
        else:
            out = subprocess.DEVNULL
        self._proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT)
        self._await_ready()
        return self

    def _await_ready(self) -> None:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise AqaError(f"vllm serve exited early (rc={self._proc.returncode}); "
                               f"see {self.log_path}")
            if self._healthy():
                _log.info("endpoint ready at %s", self.base_url)
                return
            time.sleep(2.0)
        self.stop()
        raise AqaError(f"vllm serve not ready within {self.startup_timeout}s")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
