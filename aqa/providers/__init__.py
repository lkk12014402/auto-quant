"""Model provider plugins (eval-centric core).

See :mod:`aqa.providers.base` for the abstraction.  Builtins are registered in
:data:`aqa.registry.PROVIDERS`; third parties add providers via the
``aqa.providers`` entry-point group.
"""

from __future__ import annotations

from .base import ModelProvider, ProviderContext, ResolvedModel
from .builtin import EndpointProvider, PretrainedProvider, QuantizeProvider

__all__ = [
    "EndpointProvider",
    "ModelProvider",
    "PretrainedProvider",
    "ProviderContext",
    "QuantizeProvider",
    "ResolvedModel",
]
