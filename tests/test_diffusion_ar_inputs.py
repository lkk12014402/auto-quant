"""AR-stage input wiring for the diffusion harness -- no GPU / network / vllm-omni.

Regression guard for a real bug: a model whose pipeline has a text/AR stage ahead
of the diffusion stage (HunyuanImage3) declares an ``ar_input_builder`` through
``vllm_omni.model_extras``.  The harness used to ignore it, so the AR stage never
received its ``stop_token_ids`` and decoded until ``max_tokens`` -- ~10x slower
**and** the DiT stage was conditioned on a runaway CoT instead of the recaption
prefix the model hands over by design.

These tests pin all three behaviours: a declaring model gets the AR inputs, a
non-declaring model (FLUX / Z-Image / Qwen-Image / ...) is completely unchanged,
and a tokenizer failure degrades to plain generation rather than silently
mis-terminating the AR decode.
"""

import importlib.machinery
import sys
import types
from pathlib import Path
from typing import ClassVar

from aqa.eval.harness.diffusion import DiffusionHarness


# ── fakes ────────────────────────────────────────────────────────────────────
class _FakeDiffusionParams:
    """Stand-in for ``OmniDiffusionSamplingParams`` (the DiT slot)."""

    def __init__(self, **kw):
        self.height = kw.get("height")
        self.width = kw.get("width")
        self.seed = kw.get("seed")
        self.num_outputs_per_prompt = 1


class _FakeArParams:
    """Stand-in for a non-diffusion (AR) stage's sampling params."""

    def __init__(self, **kw):
        self.temperature = kw.get("temperature", 0.0)
        self.max_tokens = kw.get("max_tokens", 8192)
        self.stop_token_ids: list[int] = list(kw.get("stop_token_ids") or [])


class _FakeImage:
    def save(self, path):
        Path(path).write_bytes(b"fake-png")


class _FakeOutput:
    images: ClassVar[list] = [_FakeImage()]


class _FakeOmni:
    instances: ClassVar[list] = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        # Two-stage topology: index 0 = AR, index 1 = DiT (HunyuanImage-3.0's
        # documented layout; the builder names the indices explicitly).
        self.default_sampling_params_list = [_FakeArParams(), _FakeDiffusionParams()]
        self.generate_calls = []
        _FakeOmni.instances.append(self)

    def generate(self, prompt, sampling_params_list=None):
        self.generate_calls.append((dict(prompt), sampling_params_list))
        return [_FakeOutput()]


class _ArInputs:
    """Stand-in for ``vllm_omni.model_extras.hunyuan_image3.ARStageInputs``."""

    def __init__(self, *, prompt_token_ids=None, prompt=None, use_system_prompt=None,
                 stop_token_ids=(), modalities=("image",), stage_indices=(0,)):
        self.prompt_token_ids = prompt_token_ids
        self.prompt = prompt
        self.use_system_prompt = use_system_prompt
        self.stop_token_ids = list(stop_token_ids)
        self.modalities = list(modalities)
        self.stage_indices = list(stage_indices)


def _install_fake_vllm_omni(monkeypatch, *, builder, model_class_name="SomeModel"):
    """Inject a fake ``vllm_omni`` whose ``model_extras`` declares *builder*."""
    _FakeOmni.instances = []
    pkg = types.ModuleType("vllm_omni")
    pkg.__spec__ = importlib.machinery.ModuleSpec("vllm_omni", loader=None)
    omni_mod = types.ModuleType("vllm_omni.entrypoints.omni")
    omni_mod.Omni = _FakeOmni
    data_mod = types.ModuleType("vllm_omni.inputs.data")
    data_mod.OmniDiffusionSamplingParams = _FakeDiffusionParams
    extras_mod = types.ModuleType("vllm_omni.model_extras")
    extras_mod.get_ar_input_builder = lambda name: builder
    extras_mod.get_model_class_name = lambda omni: model_class_name
    for name, mod in {
        "vllm_omni": pkg,
        "vllm_omni.entrypoints": types.ModuleType("vllm_omni.entrypoints"),
        "vllm_omni.entrypoints.omni": omni_mod,
        "vllm_omni.inputs": types.ModuleType("vllm_omni.inputs"),
        "vllm_omni.inputs.data": data_mod,
        "vllm_omni.model_extras": extras_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return _FakeOmni


def _install_fake_tokenizer(monkeypatch, *, raises=False):
    """Fake ``transformers.AutoTokenizer`` (heavy dependency; task-owned seam)."""
    transformers = types.ModuleType("transformers")

    class _Tok:
        name_or_path = "fake"

        @classmethod
        def from_pretrained(cls, *a, **kw):
            if raises:
                raise OSError("no tokenizer here")
            return cls()

    transformers.AutoTokenizer = _Tok
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _run_one(fake_omni, harness, prompt="a red cube"):
    omni = harness._load_omni("org/model")
    harness._generate_one(omni, prompt, 0, 42, Path("/tmp/ar-inputs-test.png"))
    return omni.generate_calls[-1]


# ── tests ────────────────────────────────────────────────────────────────────
def test_declaring_model_gets_ar_inputs(monkeypatch):
    """A model that declares a builder gets prompt tokens + AR stop tokens."""
    stop = [128044, 128045]

    def builder(*, prompt, tokenizer, extra_body, **kw):
        assert prompt == "a red cube"
        assert tokenizer is not None, "the harness must load the AR tokenizer"
        assert extra_body == {}, "no extra body means the model's own defaults"
        assert kw["num_images"] == 0, "text-to-image carries no reference images"
        assert kw["height"] == 1024 and kw["width"] == 1024
        return _ArInputs(
            prompt_token_ids=[11, 22, 33],
            use_system_prompt="en_unified",
            stop_token_ids=stop,
        )

    _install_fake_vllm_omni(monkeypatch, builder=builder)
    _install_fake_tokenizer(monkeypatch)
    harness = DiffusionHarness(omni_kwargs={"trust_remote_code": True})

    request, params = _run_one(_FakeOmni, harness)

    # request carries the AR's token ids, system prefix and modalities
    assert request["prompt_token_ids"] == [11, 22, 33]
    assert request["use_system_prompt"] == "en_unified"
    assert request["modalities"] == ["image"]
    # AR slot (index 0) gets the terminator; DiT slot (index 1) is untouched
    assert params[0].stop_token_ids == stop
    assert getattr(params[1], "stop_token_ids", None) is None


def test_non_declaring_model_is_unchanged(monkeypatch):
    """A model with no declared builder (FLUX / Z-Image / ...) is untouched."""
    _install_fake_vllm_omni(monkeypatch, builder=None)
    _install_fake_tokenizer(monkeypatch)
    harness = DiffusionHarness()

    request, params = _run_one(_FakeOmni, harness)

    assert request == {"prompt": "a red cube", "modalities": ["image"]}
    assert params[0].stop_token_ids == []
    assert getattr(params[1], "stop_token_ids", None) is None


def test_tokenizer_failure_degrades_to_plain_generation(monkeypatch):
    """No AR tokenizer -> skip the AR inputs rather than mis-terminate the decode."""
    calls = []

    def builder(**kw):
        calls.append(kw)
        return _ArInputs(stop_token_ids=[1])

    _install_fake_vllm_omni(monkeypatch, builder=builder)
    _install_fake_tokenizer(monkeypatch, raises=True)
    harness = DiffusionHarness()

    request, params = _run_one(_FakeOmni, harness)

    assert calls == [], "the builder must not be called without a tokenizer"
    assert request == {"prompt": "a red cube", "modalities": ["image"]}
    assert params[0].stop_token_ids == []


def test_builder_is_resolved_once_per_harness(monkeypatch):
    """The builder/tokenizer lookup is cached, not repeated per prompt."""
    lookups = []

    def get_builder(name):
        lookups.append(name)
        return lambda **kw: _ArInputs(prompt_token_ids=[1], stop_token_ids=[2])

    _install_fake_vllm_omni(monkeypatch, builder=None)
    monkeypatch.setattr(
        sys.modules["vllm_omni.model_extras"], "get_ar_input_builder", get_builder
    )
    _install_fake_tokenizer(monkeypatch)
    harness = DiffusionHarness()

    omni = harness._load_omni("org/model")
    for idx in range(3):
        harness._generate_one(omni, "p", idx, 42, Path(f"/tmp/ar-cache-{idx}.png"))

    assert lookups == ["SomeModel"], f"expected one lookup, got {lookups}"


def test_ar_extra_body_is_forwarded(monkeypatch):
    """``ar_extra_body`` reaches the builder (model-specific prompt controls)."""
    seen = {}

    def builder(*, extra_body, **kw):
        seen.update(extra_body)
        return _ArInputs(prompt_token_ids=[1], stop_token_ids=[2])

    _install_fake_vllm_omni(monkeypatch, builder=builder)
    _install_fake_tokenizer(monkeypatch)
    harness = DiffusionHarness(
        ar_extra_body={"bot_task": "think_recaption", "use_system_prompt": "en_unified"}
    )

    _run_one(_FakeOmni, harness)

    assert seen == {"bot_task": "think_recaption", "use_system_prompt": "en_unified"}
