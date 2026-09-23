"""PaddleOCR 2.x/3.x adapter shared by the text-rendering scorers.

CVTG-2K and GlyphDraw2 both score rendered text with PP-OCR, and both were
written against the PaddleOCR 2.x API.  The two API generations differ in

* construction -- 2.x took ``use_angle_cls``/``show_log``; 3.x takes the
  ``use_doc_orientation_classify``/``use_doc_unwarping``/
  ``use_textline_orientation`` switches plus ``enable_mkldnn`` and ``device``,
* the call -- ``ocr(img, cls=True)`` vs ``predict(img, ...)``, and
* the result shape -- a nested list of ``[box, (text, score)]`` vs a list of
  dict pages carrying ``rec_texts``.

The 3.x path additionally works around a paddlepaddle 3.3.1 bug: its default
CPU predictor runs in ``mkldnn`` mode, which crashes at inference time in
``onednn_instruction.cc`` with ``NotImplementedError:
ConvertPirAttribute2RuntimeAttribute not support``.  Paddle's own environment
switches do not influence it (``FLAGS_use_mkldnn=0`` and
``FLAGS_enable_pir_api=0`` were both tried and still crashed); the lever that
works is PaddleX's ``enable_mkldnn``, which selects ``run_mode=paddle`` instead.
The predictor is therefore built with the PaddleX default first and rebuilt once
with ``enable_mkldnn=False`` if that crash is observed, so hosts where MKL-DNN
works keep the faster path.

Language tags are translated too: the datasets label Chinese as ``zh`` while
PaddleOCR 3.x only knows ``ch`` (see :func:`paddle_lang`), and an unknown tag
raises instead of silently recognizing nothing.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ..logging_util import AqaError, get_logger

_log = get_logger("eval.paddleocr")

#: PaddleOCR 3.x construction keywords.  The 2.x call ran detection +
#: recognition only, so the extra sub-models stay off (they would otherwise be
#: downloaded and run for every image).
PADDLEOCR_3X_KWARGS = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
}


#: PaddleOCR's own language codes, keyed by the dataset tags the scorers use.
#: PaddleOCR 3.x rejects ISO-style ``zh`` outright ("No models are available for
#: lang='zh'"), and its Chinese models are registered as ``ch`` -- so a scorer
#: that passes a dataset's ``zh`` label straight through silently gets zero
#: recognized lines (a 0.0000 score that looks like a model failure).
PADDLE_LANG_ALIASES = {
    "zh": "ch",
    "zh-cn": "ch",
    "zh-hans": "ch",
    "cn": "ch",
    "zh-hant": "chinese_cht",
    "zh-tw": "chinese_cht",
    "zh-hk": "chinese_cht",
    "chinese_traditional": "chinese_cht",
    "ja": "japan",
    "jp": "japan",
    "ko": "korean",
    "kr": "korean",
    "fr": "fr",
    "de": "german",
    "es": "es",
    "ru": "ru",
}


def paddle_lang(tag: str | None) -> str:
    """Map a dataset language tag onto a PaddleOCR language code."""
    if not tag:
        return "en"
    normalized = str(tag).strip().lower().replace("_", "-")
    return PADDLE_LANG_ALIASES.get(normalized, normalized)


def import_paddleocr(component: str) -> Any:
    """``paddleocr.PaddleOCR``, or a clear :class:`AqaError` when missing."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise AqaError(
            f"{component} text scoring needs PaddleOCR: pip install paddleocr "
            "(and a paddlepaddle build for your platform)"
        ) from exc
    return PaddleOCR


def major_version() -> int:
    """Major version of the installed ``paddleocr`` (``0`` when unknown)."""
    try:
        raw = getattr(importlib.import_module("paddleocr"), "__version__", None)
    except ImportError:  # pragma: no cover - import_paddleocr raises first
        return 0
    try:
        return int(str(raw).split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def is_mkldnn_failure(exc: BaseException) -> bool:
    """True for the paddle oneDNN/PIR crash that ``enable_mkldnn=False`` fixes."""
    if isinstance(exc, NotImplementedError):
        return True
    # paddle spells it "oneDNN" in some messages and "MKL-DNN"/"mkldnn" in others
    message = str(exc).lower().replace("-", "").replace("_", "")
    return "onednn" in message or "mkldnn" in message


def paddle_device(device: str | None) -> str | None:
    """Translate a torch-style device into PaddleX's ``device`` argument.

    ``None`` means "leave the choice to PaddleX" (its own default is GPU when
    available, CPU otherwise).
    """
    if not device or device == "auto":
        return None
    name, _, index = str(device).partition(":")
    if name == "cpu":
        return "cpu"
    if name in ("cuda", "gpu"):
        return f"gpu:{index}" if index.isdigit() else "gpu"
    return None  # npu/xpu/mps/...: not ours to guess


def paddle_input(image: Any) -> Any:
    """PaddleOCR takes a path or an ndarray -- never the ``repr`` of one.

    PIL images must be converted: PaddleX 3.x *ignores* other types with a
    warning ("Only ``numpy.ndarray`` and ``str`` are supported! So has been
    ignored"), which silently yields zero recognized text -- i.e. a 0 accuracy
    that looks like a model failure.
    """
    if isinstance(image, (str, Path)):
        return str(image)
    import numpy as np

    if isinstance(image, np.ndarray):
        return image
    # PIL.Image exposes ``__array_interface__``; torch tensors expose ``__array__``.
    if hasattr(image, "__array__") or hasattr(image, "__array_interface__"):
        return np.asarray(image)
    return image


def page_rec_texts(page: Any) -> list[str] | None:
    """``rec_texts`` of a PaddleOCR 3.x page, or ``None`` for a 2.x page."""
    if isinstance(page, dict):
        texts = page.get("rec_texts")
    else:
        texts = getattr(page, "rec_texts", None)
    return None if texts is None else [str(text) for text in texts]


def result_lines(result: Any) -> list[str]:
    """Text lines of a PaddleOCR 3.x (dict pages) or 2.x (nested list) result."""
    pages = result if isinstance(result, (list, tuple)) else [result]
    lines: list[str] = []
    for page in pages:
        texts = page_rec_texts(page)
        if texts is not None:
            lines.extend(texts)
            continue
        for line in page or []:  # 2.x: [[ [box, (text, score)], ... ]]
            try:
                lines.append(line[1][0])
            except (IndexError, TypeError):
                continue
    return lines


class PaddleOCREngine:
    """PP-OCR detection + recognition, with a per-language engine cache.

    Engines are built lazily on first use; a missing ``paddleocr`` surfaces as a
    clear :class:`AqaError`.  Handles both the 2.x and 3.x API and result
    shapes, and retries once with ``enable_mkldnn=False`` if paddle's oneDNN
    backend crashes (see :func:`is_mkldnn_failure`).
    """

    def __init__(
        self, component: str = "text", lang: str = "en", device: str = "auto"
    ):
        self.component = component
        self.lang = lang
        self.device = device
        self._engines: dict[str, Any] = {}
        self._mkldnn_off: set[str] = set()

    def _build(self, lang: str, *, enable_mkldnn: bool) -> Any:
        PaddleOCR = import_paddleocr(self.component)
        code = paddle_lang(lang)
        third_party = dict(PADDLEOCR_3X_KWARGS)
        third_party["enable_mkldnn"] = enable_mkldnn
        device = paddle_device(self.device)
        if device is not None:
            third_party["device"] = device
        legacy = {"use_angle_cls": True, "lang": code, "show_log": False}
        if major_version() == 2:  # pragma: no cover - PaddleOCR 2.x
            engine = PaddleOCR(**legacy)
        else:
            try:
                engine = PaddleOCR(lang=code, **third_party)
            except TypeError:  # pragma: no cover - 2.x with an unknown version
                engine = PaddleOCR(**legacy)
            except Exception as exc:
                # e.g. "No models are available for lang='xx'": a misconfigured
                # language is an environment problem -- surface it as AqaError so
                # the caller marks the image unscored instead of scoring it 0.0.
                raise AqaError(
                    f"{self.component}: PaddleOCR cannot handle lang={lang!r} "
                    f"(mapped to {code!r}): {exc}"
                ) from exc
        self._engines[lang] = engine
        if not enable_mkldnn:
            self._mkldnn_off.add(lang)
        return engine

    def engine(self, lang: str) -> Any:
        """The (cached) engine for ``lang``, built with PaddleX defaults."""
        cached = self._engines.get(lang)
        if cached is not None:
            return cached
        return self._build(lang, enable_mkldnn=True)

    def _predict(self, engine: Any, image: Any) -> Any:
        data = paddle_input(image)
        if hasattr(engine, "predict"):  # PaddleOCR 3.x
            return engine.predict(data, **PADDLEOCR_3X_KWARGS)
        return engine.ocr(data, cls=True)  # pragma: no cover - PaddleOCR 2.x

    def recognize(self, image: Any, lang: str | None = None) -> list[str]:
        """Text lines recognized in ``image`` (a path or an ndarray)."""
        lang = lang or self.lang
        engine = self.engine(lang)
        try:
            result = self._predict(engine, image)
        except Exception as exc:  # paddle raises backend-specific exception types
            if lang in self._mkldnn_off or not is_mkldnn_failure(exc):
                raise
            _log.warning(
                "%s: PaddleOCR oneDNN backend failed (%s); retrying with MKL-DNN off",
                self.component,
                exc,
            )
            engine = self._build(lang, enable_mkldnn=False)
            result = self._predict(engine, image)
        return result_lines(result)
