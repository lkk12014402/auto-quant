"""LocalExecutor: run the quantize -> evaluate -> report pipeline locally.

The default ``executor.env.type: current`` runs in-process -- the reference
executor that lets the framework run on any machine that can already reach the
GPU (or inside a container), with no reservation system or multi-hop SSH -- the
open-source baseline path.  ``env.type: venv`` / ``docker`` instead stage an
inner ``aqa run`` inside the declared environment (see ``local_env.py``), the
same trick as the ssh executor but without leaving the machine.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..logging_util import get_logger
from ..pipeline import run_pipeline
from . import ProcessState, RunResult

_log = get_logger("executor.local")
_META = "aqa_run.json"


class LocalExecutor:
    name = "local"

    def __init__(self, **options):
        self.options = options

    def run(self, run_cfg, *, dry_run: bool = False, resume: bool = False) -> RunResult:
        from . import local_env

        local_env.check_required_env(run_cfg)  # fast-fail preflight, every env mode
        out_dir = Path(run_cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if run_cfg.output.unique_dir:
            # Convenience symlink: <base>/latest always points at the newest run dir.
            try:
                link = out_dir.parent / "latest"
                link.unlink(missing_ok=True)
                link.symlink_to(out_dir.name)
            except OSError as exc:
                _log.warning("could not update latest symlink: %s", exc)
        env_type = run_cfg.executor.env.type
        if env_type != "current":
            # venv/docker: stage an inner `aqa run` (env block dropped, so no
            # recursion) inside the requested environment; the child maintains
            # aqa_run.json itself and the outer result reports rc + that file.
            out_dir = Path(run_cfg.output.dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if env_type == "venv":
                return local_env.run_venv(run_cfg, out_dir, dry_run=dry_run)
            return local_env.run_docker(run_cfg, out_dir, dry_run=dry_run)
        out_dir = Path(run_cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / _META
        meta_path.write_text(json.dumps({"model": run_cfg.model, "state": "running"}))
        if dry_run:
            _log.info("[dry-run] would run pipeline for %s in %s", run_cfg.model, out_dir)
            return RunResult(output_dir=out_dir, state=ProcessState.DONE)
        state = run_pipeline(run_cfg, out_dir)
        meta_path.write_text(json.dumps({"model": run_cfg.model, "state": state.value}))
        return RunResult(output_dir=out_dir, state=state)

    def status(self, output_dir: Path) -> ProcessState:
        meta = Path(output_dir) / _META
        if not meta.exists():
            return ProcessState.UNKNOWN
        return ProcessState(json.loads(meta.read_text()).get("state", "unknown"))

    def stop(self, output_dir: Path) -> bool:
        # env-dispatched (venv/docker) runs are child subprocesses of aqa itself;
        # there is still nothing out-of-process to signal here.
        return False  # in-process; nothing to signal

    @staticmethod
    def detect(output_dir: Path) -> bool:
        return (Path(output_dir) / _META).exists()
