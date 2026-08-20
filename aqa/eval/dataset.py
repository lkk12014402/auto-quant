"""Dataset loading for the native harness.

Parses NeMo-Evaluator-style dataset URIs (``hf://org/name?split=test&name=cfg``)
and yields samples as plain dicts.  The ``datasets`` library is an optional
dependency; importing lazily keeps the core install slim.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..logging_util import AqaError


@dataclass
class DatasetRef:
    scheme: str
    path: str
    split: str
    name: str | None = None

    @classmethod
    def parse(cls, uri: str) -> DatasetRef:
        parsed = urlparse(uri)
        if parsed.scheme not in ("hf", "file"):
            raise AqaError(f"unsupported dataset URI scheme {parsed.scheme!r} in {uri!r}")
        query = parse_qs(parsed.query)
        path = (parsed.netloc + parsed.path).strip("/")
        return cls(
            scheme=parsed.scheme,
            path=path,
            split=query.get("split", ["test"])[0],
            name=query.get("name", [None])[0],
        )


def load_samples(uri: str, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield dataset rows as dicts. Requires the optional ``datasets`` package."""
    ref = DatasetRef.parse(uri)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dep
        raise AqaError(
            "the native harness needs the `datasets` package: pip install datasets"
        ) from exc
    if ref.scheme == "hf":
        ds = load_dataset(ref.path, ref.name, split=ref.split)
    else:  # file
        ds = load_dataset("json", data_files=ref.path, split=ref.split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield dict(row)


def render_prompt(template: str, sample: dict[str, Any]) -> str:
    """Render a ``{field}`` prompt template against a sample row."""
    try:
        return template.format(**sample)
    except KeyError as exc:
        raise AqaError(f"prompt template references missing field {exc} in sample") from exc
