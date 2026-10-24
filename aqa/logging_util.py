"""Shared logging + error types for autoquant-agent."""

from __future__ import annotations

import logging
import sys


class AqaError(Exception):
    """Base class for all autoquant-agent errors."""


class ConfigError(AqaError):
    """Raised when a run configuration is invalid."""


class RegistryError(AqaError):
    """Raised when a plugin cannot be resolved from a registry."""


class RecipeError(AqaError):
    """Raised when a benchmark / quant recipe is malformed or missing."""


_CONFIGURED = False


def get_logger(name: str = "aqa") -> logging.Logger:
    """Return the shared logger, configuring a stderr handler once."""
    global _CONFIGURED
    logger = logging.getLogger("aqa")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED = True
    return logger if name == "aqa" else logger.getChild(name)


def attach_run_log(out_dir: "str | object") -> "logging.Handler | None":
    """Tee every ``aqa.*`` log record to ``<out_dir>/logs/pipeline.log``.

    Gives each run an always-present, human-readable per-phase execution trace
    (phase enter/done, quant/eval banners, heal notes) alongside the structured
    ``session/events.jsonl``.  Returns the handler so the caller detaches it when
    the run ends (``append`` mode preserves a resumed run's earlier log).
    """
    from pathlib import Path

    get_logger()  # ensure the base stderr handler is configured first
    log_dir = Path(out_dir) / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    except OSError:
        return None
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    logging.getLogger("aqa").addHandler(handler)
    return handler


def detach_run_log(handler: "logging.Handler | None") -> None:
    """Flush, close and remove a handler returned by :func:`attach_run_log`."""
    if handler is None:
        return
    try:
        handler.flush()
        handler.close()
    finally:
        logging.getLogger("aqa").removeHandler(handler)
