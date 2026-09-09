"""Declarative run configuration for autoquant-agent.

A single ``run.yaml`` describes an entire quantize -> evaluate -> compare run,
replacing the historic ``local_dispatch`` chain of CLI flags + request.json +
container env vars + ``source config.env`` overrides with one typed object and a
single parse point.

Composition features (Hydra-inspired, no Hydra dependency), mirroring the
patterns proven by NeMo Evaluator's ``compose.py``:

* ``_base_: other.yaml``          -- inherit a parent config, override only deltas.
* ``${VAR}`` / ``${VAR:-default}`` -- environment-variable expansion in any string.
* deep-merge semantics            -- child keys win; ``null`` deletes an inherited key.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .logging_util import ConfigError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_DELETE = object()  # sentinel: an explicit ``null`` override deletes the key


# ── Sub-configs ──────────────────────────────────────────────────────────────
class QuantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheme: str = "MXFP4"
    method: Literal["model_free", "autoround"] = "model_free"
    export_format: str = "llm_compressor"
    recipe: str = "auto"  # "auto" => agent-optimize; or a path/key into recipe store
    iters: int | None = None
    nsamples: int | None = None
    ignore_layers: str | None = None
    layer_config: dict[str, Any] | None = None
    quant_lm_head: bool = False
    # --- Algorithm selection (maps to auto-round `--algorithm`) ---------------
    # Comma-separated algorithm chain, e.g. "awq", "auto_round", "rtn",
    # "hadamard", or a composition such as "awq,auto_round" (a transform plus a
    # base quant algorithm). ``None`` lets auto-round pick its default (RTN when
    # ``iters == 0`` else auto_round).
    algorithm: str | None = None
    # Enable the experimental SignRound extension ("auto-round v2"). Only has an
    # effect for the auto_round base algorithm (``--enable_alg_ext``).
    enable_alg_ext: bool = False
    # Hadamard rotation variant when ``hadamard`` is in the algorithm chain
    # (``--rotation-hadamard-type``). auto-round also auto-infers ``hadamard``
    # into the chain when this is set.
    hadamard_type: Literal["hadamard", "random_hadamard", "quarot_hadamard"] | None = None
    # RTN optimized-rounding toggle (``--enable_opt_rtn`` / ``--disable_opt_rtn``).
    opt_rtn: bool | None = None
    # Device placement for auto-round (``--device_map``). ``None`` => let the
    # pipeline pick: ``"auto"`` when the run has >1 GPU (shard the model across
    # visible cards), else auto-round's single-device default. Set explicitly
    # (e.g. ``"auto"`` / ``"0,1"``) to override the automatic choice.
    device_map: str | None = None
    # Static KV-cache FP8 (``--static_kv_dtype fp8``) and static FP8 attention
    # (``--static_attention_dtype fp8``). Both measure static scales from
    # calibration data, so they only apply to the calibration path
    # (method="autoround"); model_free never runs the model.
    static_kv_fp8: bool = False
    static_attention_fp8: bool = False
    # Multi-recipe sweep (`aqa sweep`): explore several candidate plans instead
    # of a single quantization. ``True`` uses the built-in candidate set derived
    # from the model's structure (conservative/aggressive/mixed/tuning);
    # ``recipe_sweep_candidates`` replaces it with an explicit list of
    # {name, description?, <any QuantConfig field>} entries.
    recipe_sweep: bool = False
    recipe_sweep_candidates: list[dict[str, Any]] | None = None
    extra_args: list[str] = Field(default_factory=list)

    _ALG_TOKENS = {
        "rtn",
        "auto_round",
        "autoround",
        "sign_round",
        "signround",
        "awq",
        "hadamard",
        "random_hadamard",
        "quarot_hadamard",
        "svdquant",
    }
    _AUTO_ROUND_TOKENS = {"auto_round", "autoround", "sign_round", "signround"}
    # Preprocessors/transforms may be stacked (applied left->right before the
    # terminal block-quantizer); block-quantizers are mutually exclusive.
    _PREPROCESSOR_TOKENS = {
        "awq",
        "hadamard",
        "random_hadamard",
        "quarot_hadamard",
        "svdquant",
    }

    @model_validator(mode="after")
    def _validate_algorithm(self) -> QuantConfig:
        tokens: list[str] = []
        if self.algorithm:
            tokens = [t.strip().lower() for t in self.algorithm.split(",") if t.strip()]
            unknown = [t for t in tokens if t not in self._ALG_TOKENS]
            if unknown:
                raise ValueError(
                    f"unknown quant algorithm(s) {unknown}; "
                    f"choose from {sorted(self._ALG_TOKENS)} (comma-separated)"
                )
            # auto-round allows many preprocessors but exactly one block-quantizer.
            block_kinds = set()
            if "rtn" in tokens:
                block_kinds.add("rtn")
            if set(tokens) & self._AUTO_ROUND_TOKENS:
                block_kinds.add("auto_round")
            if len(block_kinds) > 1:
                raise ValueError(
                    "only one block-quantizer is allowed; 'rtn' and 'auto_round' "
                    "are mutually exclusive (preprocessors awq/hadamard/svdquant "
                    "may be stacked with a single quantizer, e.g. 'awq,auto_round')"
                )
            # model_free is calibration-free (RTN only); anything that needs
            # calibration (auto_round tuning or a preprocessor) contradicts it.
            if self.method == "model_free" and any(t != "rtn" for t in tokens):
                raise ValueError(
                    "method='model_free' is calibration-free and only supports "
                    "algorithm='rtn'; set method='autoround' to use "
                    f"{sorted(t for t in tokens if t != 'rtn')}"
                )
        if self.enable_alg_ext and tokens and not (set(tokens) & self._AUTO_ROUND_TOKENS):
            raise ValueError(
                "enable_alg_ext (auto-round v2) only applies to the auto_round "
                "algorithm; add 'auto_round' to `algorithm` or drop enable_alg_ext"
            )
        if self.method == "model_free" and (self.static_kv_fp8 or self.static_attention_fp8):
            raise ValueError(
                "static_kv_fp8 / static_attention_fp8 need calibration (static "
                "scales are measured from data); set method='autoround' to use them"
            )
        if self.recipe_sweep_candidates and not self.recipe_sweep:
            raise ValueError(
                "recipe_sweep_candidates is set but recipe_sweep is false; "
                "set recipe_sweep: true and run with `aqa sweep`")
        return self


class ModelUploadConfig(BaseModel):
    """Upload the quantized checkpoint to a Hugging Face *model* repo after a
    successful run.  Minimal single-token/single-org port (no quota ledger)."""

    model_config = ConfigDict(extra="forbid")

    repo: str  # org/owner to create the model repo under
    repo_name: str | None = None  # repo id under `repo`; default = artifact name
    token_env: str | None = None
    private: bool = False
    endpoint: str | None = None
    disable_xet: bool = False
    model_card: bool = True


class ProviderConfig(BaseModel):
    """How the model-under-eval is produced/served (the eval-centric core).

    A *provider* answers "where does the model being evaluated come from":

    * ``pretrained`` -- a HF repo id or local path, evaluated as-is (default).
    * ``quantize``   -- run autoquant (optionally agent-optimize + fix-loop) to
      produce a quantized checkpoint, then evaluate that. Carries the historic
      :class:`QuantConfig` fields, so quantization is now a *pluggable provider*
      rather than the framework's reason for being.
    * ``endpoint``   -- an already-served OpenAI-compatible endpoint (agentic
      evals drive it directly; no local load).

    Extra keys are forwarded to third-party provider plugins.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "pretrained"
    # `quantize` provider fields (mirror the legacy top-level `quant:` block).
    quant: QuantConfig | None = None
    # Optional: push the quantized checkpoint to a HF model repo on success.
    upload: ModelUploadConfig | None = None
    # `endpoint` provider fields.
    base_url: str | None = None
    api_key_env: str | None = None
    served_model: str | None = None


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_accuracy_drop: float | None = None  # quantized vs baseline; None => no gate
    min_accuracy: float | None = None
    significance: bool = False  # require the drop to be statistically significant
    alpha: float = 0.05  # significance level for the paired/unpaired test


class LmEvalOptions(BaseModel):
    """Harness-specific knobs for the ``lm_eval`` backend.

    ``model_args`` are extra ``key=value`` pairs appended to lm_eval's
    ``--model_args`` (e.g. ``enable_thinking: false``, ``reasoning_parser: qwen3``,
    ``max_num_seqs: 128``) — user overrides win over the backend defaults.
    ``group`` merges benchmarks that share the same generation parameters into a
    single ``lm_eval --tasks a,b,c`` call (e.g. piqa+hellaswag+mmlu), while tasks
    needing distinct params (e.g. gsm8k) run separately.
    """

    model_config = ConfigDict(extra="forbid")

    model_args: dict[str, Any] = Field(default_factory=dict)
    gen_kwargs: dict[str, Any] = Field(default_factory=dict)  # default --gen_kwargs (generative)
    apply_chat_template: bool | None = None  # global override (None => per-recipe)
    fewshot_as_multiturn: bool | None = None
    seed: int = 42
    group: bool = True  # merge compatible tasks into one lm_eval invocation


class DiffusionEvalOptions(BaseModel):
    """Harness-specific knobs for the ``diffusion`` (vllm-omni text-to-image) harness.

    The generation parameters double as the recipe-grouping key: benchmarks that
    share them are served by a single ``Omni`` model load (see
    :meth:`DiffusionHarness.run_group`).  ``scorer_model`` is the CLIP checkpoint
    used for the image-text score; ``prompt_field``/``dataset_split`` tell the
    harness where to find prompts in the recipe's ``dataset``.
    """

    model_config = ConfigDict(extra="forbid")

    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    height: int = 1024
    width: int = 1024
    seed: int = 42
    scorer_model: str = "openai/clip-vit-base-patch32"
    scorer_device: str = "auto"  # "auto" => cuda when available, else cpu
    prompt_field: str = "Prompts"  # default prompt column; recipe.prompt_field wins
    dataset_split: str = "train"
    # Extra kwargs forwarded verbatim to ``Omni(model=..., mode="text-to-image", ...)``
    # — per-model engine knobs such as enforce_eager / enable_cpu_offload /
    # cache_backend. Empty for the common case.
    omni_kwargs: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmarks: list[str] = Field(default_factory=lambda: ["piqa", "mmlu", "hellaswag"])
    backend: str = "hf"  # hf | vllm | vllm_xpu | vllm_omni
    harness: str = "lm_eval"  # default harness for bare tasks: lm_eval | evalscope | native
    batch_size: str = "auto"
    baseline: bool = False  # also evaluate the un-quantized bf16 model
    limit: int | None = None  # cap samples per benchmark (smoke runs)
    generative_tasks: list[str] = Field(default_factory=lambda: ["gsm8k"])
    gen_max_toks: int = 2048

    # Thinking-mode evaluation. Reasoning models (qwen3, glm, ...) answer very
    # differently with chain-of-thought on vs off, so both settings may need
    # measuring. ``auto`` (default) detects a thinking-capable model from its
    # chat template (`enable_thinking` switch) and runs BOTH modes when found —
    # single run otherwise (identical to legacy behavior). ``both`` forces the
    # dual run, ``on``/``off`` force a single mode. ``thinking_off_model_args``
    # is merged into the backend's model_args for the "off" run; override it
    # for models whose disable-switch differs from qwen-style enable_thinking.
    thinking: Literal["auto", "both", "on", "off"] = "auto"
    thinking_off_model_args: dict[str, Any] = Field(
        default_factory=lambda: {"enable_thinking": False})

    # Per-harness option blocks (differentiate the eval methods under `eval`).
    lm_eval: LmEvalOptions = Field(default_factory=LmEvalOptions)
    diffusion: DiffusionEvalOptions = Field(default_factory=DiffusionEvalOptions)
    evalscope: dict[str, Any] = Field(default_factory=dict)  # passthrough for the plugin


class PatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture: bool = True
    review: bool = False
    track_vllm: bool = True


class KnowledgeConfig(BaseModel):
    """Shared, persistent fix-knowledge store (lessons + replayable patches).

    Lessons and successful patches are keyed by error *signature* so they are
    reused across runs, models and machines -- not scoped to one run's output.
    When ``repo`` is set the store is backed by a Hugging Face dataset (synced
    down at run start, pushed back after a successful fix); otherwise it is a
    local-only directory shared across runs on the same host.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    repo: str | None = None          # HF dataset repo, e.g. "org/aqa-knowledge"
    path_in_repo: str = "knowledge"  # subtree inside the dataset
    token_env: str | None = None     # env var holding the write token (else HF_TOKEN)
    endpoint: str | None = None      # custom HF endpoint (mirrors the sink)
    local_dir: str | None = None     # local cache; default ~/.cache/aqa/knowledge
    replay: bool = True              # try recorded patches before invoking the agent
    push: bool = True                # upload new lessons/patches back to the repo
    # Whether the optimize agent reads the synced `references/` layer (fresher,
    # owner-maintained) or is pinned to the packaged copies. Refreshing is
    # always a manual owner action (aqa knowledge push-references); this only
    # gates consumption. Set false for pinned/reproducible experiments.
    use_references: bool = True


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fix_loop: bool = True
    optimize: bool = False  # agent-optimize: analyse model -> produce quant recipe
    tiers: list[str] = Field(default_factory=lambda: ["openclaw", "minimax", "opus"])
    max_fix_attempts: int = 5
    tier_attempt_budget: int = 3  # attempts on a tier before fail-safe escalation
    drift_threshold: int = 2  # same error_class N times in a row -> escalate
    agent_timeout: int = 1200  # per-attempt agent wall-clock budget (seconds)
    context: str | None = None  # inline text or "@file" (resolved by loader)
    patch: PatchConfig = Field(default_factory=PatchConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    # End-of-run session sedimentation: ATIF-ize the run's fix-loop/optimize
    # sessions (+ anything the owner dropped into <run_dir>/sessions/) and push
    # to the results dataset's trajectories/ tree. Best-effort, never blocks.
    sediment_sessions: bool = True


class SshAuth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["password", "key", "agent"] = "agent"
    password_env: str | None = None  # secrets are ONLY referenced via env, never inline
    key_file: str | None = None


class SshHop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str
    auth: SshAuth = Field(default_factory=SshAuth)


class SshConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str
    gpus: str = "0"
    # Accelerator family of the remote host.  "cuda" => NVIDIA (`--gpus device=`);
    # "xpu" => Intel GPU (`/dev/dri` passthrough + `ZE_AFFINITY_MASK`, Intel vllm
    # base image).  Selects both the device flags and the default base image.
    accelerator: Literal["cuda", "xpu"] = "cuda"
    auth: SshAuth = Field(default_factory=SshAuth)
    jump: list[SshHop] = Field(default_factory=list)
    # Extra bind mounts for the remote container (docker `-v host:cont` specs,
    # e.g. a shared filesystem holding local model weights:
    # ["/home:/home"] makes /home/user/models visible inside the run).
    mounts: list[str] = Field(default_factory=list)


class ImageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu: str | None = None
    xpu: str | None = None
    # "local" => build an independent Dockerfile shipped with the repo on top of
    # `gpu`/`xpu` as the base image; a path => use that Dockerfile; None => run the
    # base image directly. Mirrors local_dispatch's `--gpu-dockerfile local`.
    dockerfile: str | None = None
    build_args: dict[str, str] = Field(default_factory=dict)
    # Extra pip packages installed inside the container by the bootstrap, in
    # addition to the auto-derived stack (aqa + auto-round + lm-eval, plus vllm
    # when the eval backend is vllm). Use to pin versions or add flashinfer, etc.
    pip_install: list[str] = Field(default_factory=list)


class ProxyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: str | None = None


class EnvConfig(BaseModel):
    """Which local environment ``executor.type: local`` runs the pipeline in.

    ``current`` (default) is the historical in-process behavior.  ``venv``
    stages an inner ``aqa run`` inside a (possibly provisioned) virtualenv at
    ``path``; ``docker`` builds/runs a local container (image config stays in
    ``executor.image``).  Both mirror the ssh executor's staged-run flow: the
    inner config forces a plain ``type: local`` executor (env block dropped),
    so there is no dispatch recursion.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["current", "venv", "docker"] = "current"
    path: str | None = None  # venv: the venv directory (provisioned when missing + create)
    packages: str | list[str] = "auto"  # "auto" => derive from run config; or explicit list
    create: bool = True  # venv: provision <path> (python3 -m venv + pip install) when missing
    mounts: list[str] = Field(default_factory=list)  # docker: extra `-v host:cont` bind mounts
    # docker: `--network` mode. Host networking is the sane default for local GPU
    # runs (the bridge has no internet on typical hosts, and pip installs/HF
    # downloads inside the container need it; serving ports also stay reachable).
    network: str | None = "host"

    @model_validator(mode="after")
    def _check(self) -> EnvConfig:
        if self.type == "venv" and not self.path:
            raise ConfigError("executor.env.type=venv requires `env.path` (the venv directory)")
        if isinstance(self.packages, str) and self.packages != "auto":
            raise ConfigError(
                "executor.env.packages must be \"auto\" or a list of packages, "
                f"got {self.packages!r}"
            )
        return self


class ExecutorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "local"  # local | ssh | intel_reserve (plugin) | <your plugin>
    num_gpus: int | None = None  # explicit device count; else derived from ssh.gpus
    image: ImageConfig = Field(default_factory=ImageConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    ssh: SshConfig | None = None
    env: EnvConfig = Field(default_factory=EnvConfig)  # local executor: run environment
    # Fast-fail preflight for the local executor (every env mode): each named
    # environment variable must be set before any work starts (empty => no check).
    required_env: list[str] = Field(default_factory=list)
    machines_file: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)  # forwarded to plugin executors


class SinkConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # extra keys forwarded to the sink plugin

    type: str


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str = "./results"
    sinks: list[SinkConfig] = Field(default_factory=lambda: [SinkConfig(type="local_fs")])
    # When true, each run lands in a fresh `dir/<YYYYMMDD-HHMMSS>` subdir (with a
    # `latest` symlink repointed), so run series never mix artifacts (tonight:
    # run10/run11 shared a dir and the checkpoint provenance got ambiguous).
    # Default false: an unchanged dir keeps the resume semantics.
    unique_dir: bool = False
    # Reporting identity overrides for the dataset sink. When `model:` is a local
    # checkpoint path (no `Org/Name`), the org slug can't be inferred and the
    # dataset lands under `results/unknown/...`; likewise an eval-only run of a
    # quantized dir has no `quant` block, so the artifact name defaults to
    # `<dir>-BF16-Baseline`. Set these to pin the canonical identity:
    #   report_model : canonical `Org/Model` id -> dataset org + artifact short name
    #   artifact_name: explicit artifact name (dataset path segment + model repo)
    report_model: str | None = None
    artifact_name: str | None = None


# ── Top-level ────────────────────────────────────────────────────────────────
class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    # Eval-centric core: `provider` says how the model-under-eval is produced.
    # Legacy top-level `quant:` is folded into `provider: {type: quantize}` by the
    # loader (see `_normalize_provider`), so old run.yaml files keep working.
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    gate: GateConfig | None = None
    agent: AgentConfig = Field(default_factory=AgentConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    # Phase-hook chain (capability components; see aqa/hooks/).  None => the
    # built-in default set; an explicit list => exactly those; [] => all off.
    hooks: list[str] | None = None

    @property
    def quant(self) -> QuantConfig | None:
        """Back-compat accessor: the quantize provider's :class:`QuantConfig`.

        Returns ``None`` unless the resolved provider is ``quantize`` (so existing
        ``run_cfg.quant is None`` checks in the pipeline keep their meaning).
        """
        if self.provider.type == "quantize":
            return self.provider.quant
        return None

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        if "/" not in self.model and not Path(self.model).exists():
            raise ConfigError(
                f"model must be a HF repo id (org/name) or a local path, got {self.model!r}"
            )
        if self.provider.type == "quantize" and self.provider.quant is None:
            self.provider.quant = QuantConfig()
        if self.hooks is not None:
            from .registry import HOOKS

            unknown = [h for h in self.hooks if h not in HOOKS.names()]
            if unknown:
                raise ConfigError(
                    f"unknown hook(s) {unknown}; registered: {', '.join(HOOKS.names())}"
                )
        return self


# ── Loading / composition ────────────────────────────────────────────────────
def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            var, default = m.group(1), m.group(2)
            if var in os.environ:
                return os.environ[var]
            if default is not None:
                return default
            raise ConfigError(f"environment variable ${{{var}}} is not set and has no default")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` onto ``base``; ``null`` in override deletes a key."""
    out = dict(base)
    for key, val in override.items():
        if val is None and key in out:
            out.pop(key)
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_raw(path: Path, _seen: set[Path]) -> dict:
    """Load a YAML file, resolving a single ``_base_`` chain (relative to the file)."""
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"circular _base_ reference at {path}")
    _seen.add(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    base_ref = data.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        base_data = _load_raw(base_path, _seen)
        data = _deep_merge(base_data, data)
    return data


def _resolve_context(agent: dict, base_dir: Path) -> None:
    """Resolve an ``agent.context: "@file"`` reference into inline text."""
    ctx = agent.get("context")
    if isinstance(ctx, str) and ctx.startswith("@"):
        ctx_path = (base_dir / ctx[1:]).resolve()
        if not ctx_path.exists():
            raise ConfigError(f"agent.context file not found: {ctx_path}")
        agent["context"] = ctx_path.read_text()


def _normalize_provider(raw: dict) -> dict:
    """Fold a legacy top-level ``quant:`` block into ``provider: {type: quantize}``.

    Eval-centric configs use ``provider:``; older quant-centric run.yaml files put
    ``quant:`` at the top level.  This shim keeps those working unchanged: a
    top-level ``quant`` becomes the quantize provider's config.  Raises if a config
    sets both ``quant:`` and a non-quantize ``provider:`` (ambiguous intent).
    """
    if not isinstance(raw, dict) or "quant" not in raw:
        return raw
    quant = raw.pop("quant")
    if quant is None:  # explicit `quant: null` => eval-only, just drop it.
        return raw
    provider = raw.get("provider")
    if provider is None:
        raw["provider"] = {"type": "quantize", "quant": quant}
    elif isinstance(provider, dict):
        ptype = provider.get("type", "quantize")
        if ptype != "quantize":
            raise ConfigError(
                f"config sets top-level `quant:` but provider.type={ptype!r}; "
                "move quant under `provider: {type: quantize, quant: {...}}`"
            )
        provider.setdefault("type", "quantize")
        provider.setdefault("quant", quant)
    return raw


def load_run_config(path: str | Path) -> RunConfig:
    """Load, compose, env-expand, and validate a ``run.yaml`` into a RunConfig."""
    p = Path(path)
    raw = _load_raw(p, set())
    raw = _expand_env(raw)
    raw = _normalize_provider(raw)
    if isinstance(raw.get("agent"), dict):
        _resolve_context(raw["agent"], p.resolve().parent)
    try:
        cfg = RunConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError -> ConfigError
        raise ConfigError(f"invalid run config {p}: {exc}") from exc
    if cfg.output.unique_dir:
        cfg = cfg.model_copy(update={"output": cfg.output.model_copy(
            update={"dir": _timestamped_dir(cfg.output.dir)})})
    return cfg


def _timestamped_dir(base: str) -> str:
    """Append a run-start timestamp subdir (``base/YYYYMMDD-HHMMSS``)."""
    import time

    return str(Path(base) / time.strftime("%Y%m%d-%H%M%S"))


def parse_run_config(data: dict) -> RunConfig:
    """Validate an in-memory dict (already composed) into a RunConfig."""
    return RunConfig.model_validate(_normalize_provider(_expand_env(dict(data))))
