"""PaddleOCR 2.x/3.x adapter tests -- hermetic (no paddleocr, GPU, or network).

The shared adapter is exercised at its seams: result parsing for both API
generations, device translation, and the oneDNN workaround, whose first
``predict`` is made to raise paddlepaddle 3.3.1's real crash so the retry with
``enable_mkldnn=False`` is observed for real.  A fake ``paddleocr`` module is
injected into ``sys.modules``, so nothing paddle-shaped is imported.
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import pytest

from aqa.eval import _paddleocr as po
from aqa.logging_util import AqaError

ONE_DNN_CRASH = (
    "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support "
    "[pir::ArrayAttribute<pir::DoubleAttribute>] "
    "(at /paddle/paddle/fluid/framework/new_executor/instruction/onednn/"
    "onednn_instruction.cc:116)"
)


class FakePaddleOCR:
    """Records how it was built and, on request, crashes like the oneDNN path."""

    instances: ClassVar[list[FakePaddleOCR]] = []

    def __init__(self, *, fail: bool = False, **kwargs):
        self.kwargs = kwargs
        self.calls: list[object] = []
        self.fail = fail
        FakePaddleOCR.instances.append(self)

    def predict(self, image, **kwargs):
        self.calls.append(image)
        if self.fail and len(self.calls) == 1:
            raise NotImplementedError(ONE_DNN_CRASH)
        return [
            {
                "rec_texts": ["Hello", "World"],
                "rec_scores": [0.99, 0.87],
                "enable_mkldnn": self.kwargs.get("enable_mkldnn"),
            }
        ]


@pytest.fixture(autouse=True)
def _clear_instances():
    FakePaddleOCR.instances = []
    yield
    FakePaddleOCR.instances = []


def _install(monkeypatch, *, version="3.7.0", fail_first=False):
    module = types.ModuleType("paddleocr")
    module.__version__ = version
    built = {"count": 0}

    def factory(**kwargs):
        built["count"] += 1
        return FakePaddleOCR(fail=fail_first and built["count"] == 1, **kwargs)

    module.PaddleOCR = factory
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    return module


# ── result parsing ───────────────────────────────────────────────────────────
def test_result_lines_reads_3x_dict_pages():
    result = [{"rec_texts": ["Hello", "World"], "rec_scores": [0.99, 0.87]}]
    assert po.result_lines(result) == ["Hello", "World"]


def test_result_lines_reads_3x_object_pages():
    page = types.SimpleNamespace(rec_texts=["A", "B"])
    assert po.result_lines([page]) == ["A", "B"]


def test_result_lines_reads_2x_nested_list():
    page = [[[[0, 0], [1, 0], [1, 1]], ("Hello", 0.99)],
            [[[0, 2], [1, 2], [1, 3]], ("World", 0.87)]]
    assert po.result_lines([page]) == ["Hello", "World"]


def test_result_lines_empty_page_has_no_lines():
    assert po.result_lines([{"rec_texts": [], "rec_scores": []}]) == []
    assert po.result_lines([]) == []


def test_page_rec_texts_none_for_2x_page():
    assert po.page_rec_texts([[[0, 0], [1, 0], [1, 1]], ("Hello", 0.9)]) is None


# ── device translation ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("auto", None),
        (None, None),
        ("", None),
        ("cpu", "cpu"),
        ("cuda", "gpu"),
        ("cuda:0", "gpu:0"),
        ("gpu:1", "gpu:1"),
        ("mps", None),  # not ours to guess: PaddleX picks its own default
    ],
)
def test_paddle_device_translation(device, expected):
    assert po.paddle_device(device) == expected


def test_paddle_input_keeps_arrays_and_stringifies_paths(tmp_path):
    assert po.paddle_input(str(tmp_path / "a.png")) == str(tmp_path / "a.png")
    assert po.paddle_input(tmp_path / "a.png") == str(tmp_path / "a.png")
    assert po.paddle_input(123) == 123  # anything else is passed through untouched


def test_paddle_input_converts_pil_images():
    """The harness hands the scorers PIL images; PaddleOCR silently drops those."""
    import numpy as np
    from PIL import Image

    image = Image.new("RGB", (4, 2), color="white")
    converted = po.paddle_input(image)
    assert isinstance(converted, np.ndarray)
    assert converted.shape == (2, 4, 3)

    array = np.zeros((2, 4, 3), dtype=np.uint8)
    assert po.paddle_input(array) is array


# ── mkldnn failure detection ─────────────────────────────────────────────────
def test_is_mkldnn_failure_matches_the_onednn_crash():
    assert po.is_mkldnn_failure(NotImplementedError(ONE_DNN_CRASH))
    assert po.is_mkldnn_failure(RuntimeError("MKL-DNN kernel failed"))
    assert not po.is_mkldnn_failure(ValueError("no text detected"))


# ── engine construction ──────────────────────────────────────────────────────
def test_engine_builds_3x_with_flags_and_device(monkeypatch):
    _install(monkeypatch)
    engine = po.PaddleOCREngine("cvtg2k", "en", device="cuda:0")
    assert engine.recognize("/tmp/a.png") == ["Hello", "World"]
    built = FakePaddleOCR.instances[0]
    assert built.kwargs == {
        "lang": "en",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": True,
        "device": "gpu:0",
    }
    assert built.calls == ["/tmp/a.png"]


def test_engine_caches_one_engine_per_language(monkeypatch):
    _install(monkeypatch)
    engine = po.PaddleOCREngine("glyphdraw2")
    engine.recognize("/tmp/en.png", "en")
    engine.recognize("/tmp/zh.png", "zh")
    engine.recognize("/tmp/en2.png", "en")
    assert [i.kwargs["lang"] for i in FakePaddleOCR.instances] == ["en", "ch"]  # zh -> ch
    assert len(FakePaddleOCR.instances[0].calls) == 2


def test_engine_reports_missing_paddleocr(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    engine = po.PaddleOCREngine("cvtg2k")
    with pytest.raises(AqaError, match="cvtg2k text scoring needs PaddleOCR"):
        engine.recognize("/tmp/a.png")


def test_engine_raises_other_predict_errors(monkeypatch):
    _install(monkeypatch)

    def boom(self, image, **kwargs):
        raise ValueError("no text detected")

    monkeypatch.setattr(FakePaddleOCR, "predict", boom)
    engine = po.PaddleOCREngine("glyphdraw2")
    with pytest.raises(ValueError, match="no text detected"):
        engine.recognize("/tmp/a.png")
    assert len(FakePaddleOCR.instances) == 1  # no useless rebuild


# ── the oneDNN workaround ────────────────────────────────────────────────────
def test_engine_retries_with_mkldnn_disabled(monkeypatch):
    _install(monkeypatch, fail_first=True)
    engine = po.PaddleOCREngine("cvtg2k", "en", device="cpu")
    assert engine.recognize("/tmp/a.png") == ["Hello", "World"]
    assert [i.kwargs["enable_mkldnn"] for i in FakePaddleOCR.instances] == [True, False]
    # the working engine replaces the crashing one, so the retry happens once
    assert engine.recognize("/tmp/b.png") == ["Hello", "World"]
    assert [i.kwargs["enable_mkldnn"] for i in FakePaddleOCR.instances] == [True, False]
    assert FakePaddleOCR.instances[1].calls == ["/tmp/a.png", "/tmp/b.png"]


def test_engine_does_not_retry_twice_per_language(monkeypatch):
    _install(monkeypatch, fail_first=True)
    engine = po.PaddleOCREngine("glyphdraw2")
    engine.recognize("/tmp/en.png", "en")  # en falls back to a second engine
    engine.recognize("/tmp/en2.png", "en")  # cached: no third engine
    assert [i.kwargs["enable_mkldnn"] for i in FakePaddleOCR.instances] == [True, False]
    engine.recognize("/tmp/zh.png", "zh")  # zh gets its own engine
    assert [i.kwargs["lang"] for i in FakePaddleOCR.instances] == ["en", "en", "ch"]


def test_engine_falls_back_to_2x_when_the_version_is_unknown(monkeypatch):
    _install(monkeypatch, version="unknown")

    def picky(**kwargs):
        if "use_textline_orientation" in kwargs:
            raise TypeError("unexpected keyword argument")
        return FakePaddleOCR(**kwargs)

    monkeypatch.setattr(sys.modules["paddleocr"], "PaddleOCR", picky)
    engine = po.PaddleOCREngine("cvtg2k", "en")
    assert engine.recognize("/tmp/a.png") == ["Hello", "World"]
    assert FakePaddleOCR.instances[0].kwargs == {
        "use_angle_cls": True, "lang": "en", "show_log": False,
    }


def test_engine_uses_2x_call_for_paddleocr_2(monkeypatch):
    _install(monkeypatch, version="2.7.3")

    class Legacy:
        """PaddleOCR 2.x: no ``predict``, only ``ocr(img, cls=True)``."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[object] = []

        def ocr(self, image, **kwargs):
            self.calls.append((image, kwargs))
            return [[[[[0, 0], [1, 0], [1, 1]], ("Legacy", 0.9)]]]

    monkeypatch.setattr(sys.modules["paddleocr"], "PaddleOCR", Legacy)
    engine = po.PaddleOCREngine("cvtg2k", "en")
    assert engine.recognize("/tmp/a.png") == ["Legacy"]
    legacy = engine.engine("en")
    assert legacy.kwargs == {"use_angle_cls": True, "lang": "en", "show_log": False}
    assert legacy.calls == [("/tmp/a.png", {"cls": True})]


# ── language tags ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("en", "en"),
        ("zh", "ch"),          # PaddleOCR 3.x has no 'zh' model: only 'ch'
        ("ZH_CN", "ch"),
        ("zh-Hant", "chinese_cht"),
        ("ja", "japan"),
        ("ko", "korean"),
        ("de", "german"),
        (None, "en"),
        ("", "en"),
        ("xx", "xx"),          # unknown tags pass through unchanged
    ],
)
def test_paddle_lang_mapping(tag, expected):
    assert po.paddle_lang(tag) == expected


def test_engine_builds_zh_with_paddle_code(monkeypatch):
    """A dataset 'zh' tag must reach PaddleOCR as 'ch', not raise."""
    _install(monkeypatch)
    engine = po.PaddleOCREngine("glyphdraw2")
    assert engine.recognize("/tmp/zh.png", "zh") == ["Hello", "World"]
    assert FakePaddleOCR.instances[0].kwargs["lang"] == "ch"


def test_engine_maps_unsupported_language_to_aqa_error(monkeypatch):
    """An unusable language tag must be visible, not a silently empty OCR."""
    _install(monkeypatch)

    def picky(**kwargs):
        raise ValueError(f"No models are available for lang={kwargs['lang']!r} and ocr_version=None.")

    monkeypatch.setattr(sys.modules["paddleocr"], "PaddleOCR", picky)
    engine = po.PaddleOCREngine("glyphdraw2")
    with pytest.raises(AqaError, match="cannot handle lang='xx'"):
        engine.recognize("/tmp/a.png", "xx")


# ── version detection ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("version", "expected"),
    [("3.7.0", 3), ("2.7.3", 2), ("unknown", 0), (None, 0)],
)
def test_major_version(monkeypatch, version, expected):
    module = types.ModuleType("paddleocr")
    module.__version__ = version
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    assert po.major_version() == expected


def test_major_version_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    assert po.major_version() == 0
