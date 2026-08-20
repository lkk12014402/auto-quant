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
