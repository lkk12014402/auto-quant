"""LocalExecutor: run the quantize -> evaluate -> report pipeline in-process.

This is the reference executor that lets the framework run on any machine that
can already reach the GPU (or inside a container), with no reservation system or
multi-hop SSH -- the open-source baseline path.
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
        out_dir = Path(run_cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            # Refuse loudly rather than silently starting fresh (the old no-op).
            from ..session import load_session_manifest
            load_session_manifest(out_dir / "session")  # raises AqaError if absent
        meta_path = out_dir / _META
        meta_path.write_text(json.dumps({"model": run_cfg.model, "state": "running"}))
        if dry_run:
            _log.info("[dry-run] would run pipeline for %s in %s", run_cfg.model, out_dir)
            return RunResult(output_dir=out_dir, state=ProcessState.DONE)
        try:
            state = run_pipeline(run_cfg, out_dir, resume=resume)
        except Exception:
            meta_path.write_text(
                json.dumps({"model": run_cfg.model, "state": ProcessState.FAILED.value})
            )
            raise
        meta_path.write_text(json.dumps({"model": run_cfg.model, "state": state.value}))
        return RunResult(output_dir=out_dir, state=state)

    def status(self, output_dir: Path) -> ProcessState:
        meta = Path(output_dir) / _META
        if not meta.exists():
            return ProcessState.UNKNOWN
        return ProcessState(json.loads(meta.read_text()).get("state", "unknown"))

    def stop(self, output_dir: Path) -> bool:
        return False  # in-process; nothing to signal

    @staticmethod
    def detect(output_dir: Path) -> bool:
        return (Path(output_dir) / _META).exists()
