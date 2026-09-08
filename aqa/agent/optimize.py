"""Agent-optimize: analyse a model and let an agent propose an optimal auto-round command.

Ports ``local_dispatch/agent/agent_optimize.py`` into the framework.  The flow:

  1. :func:`analyze_model` -- download-free structure analysis (config.json + the
     safetensors index *names only*, so even huge MoE models are cheap).
  2. :func:`build_optimize_prompt` -- ask the agent for ONE ``auto-round`` command.
  3. :func:`extract_command` + :func:`validate_command` -- parse the agent's answer
     and enforce safety (never FP-protect routed-expert weights; exact output dir).
  4. :func:`run_agent_optimize` -- orchestrate (analyse -> agent -> extract/validate),
     returning a validated command or ``None`` (caller falls back to the deterministic
     builder, so this can only help, never block, a run).
"""

from __future__ import annotations

import json
import re
import shlex
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..logging_util import get_logger
from .tiers import Tier

if TYPE_CHECKING:
    from ..config import QuantConfig

_log = get_logger("agent.optimize")

_LAYER_IDX_RE = re.compile(r"\.\d+\.")
_TRAIL_IDX_RE = re.compile(r"\.\d+$")
_CMD_BLOCK_RE = re.compile(r"```(?:autoround-cmd|bash|sh)?\s*\n(.*?)```", re.DOTALL)

# Structured recipe document the agent is asked to produce.  Ported from
# ``local_dispatch/agent/agent_optimize.py`` -- the readable 8-section analysis is
# what makes the recommendation auditable/intuitive (not just a bare command).
RECIPE_DOC_TEMPLATE = """# <model> AutoRound Quantization Recipe

## 1. TL;DR
- Recommended scheme / method(iters) / export format, one-line rationale each
- Expected accuracy-retention rate
- Final command: see section 6

## 2. Model structure analysis
- Architecture / model_type / parameter scale / layer count
- Dense vs MoE (expert count, experts-per-token, fused 3-D expert or not)
- Attention type: MHA / GQA / MLA(latent) / linear(Mamba) / sparse(DSA/indexer)
- Actual router/gate module names (from the safetensors index)
- Special modules: shared_experts / MTP / multimodal(vision/mm_projector) / indexer

## 3. Module-name inventory (safetensors index, de-indexed)
- Global / attention / MLP(dense) / MLP(MoE experts) / router / special prefixes

## 4. AutoRound built-in handling for this structure
- Layers this model_type auto-ignores/special-cases (e.g. `.gate` routers)

## 5. Quantization config decision
- scheme rationale (MXFP4 vs MXFP4_RCEIL vs MXFP8 vs W4A16/W8A16; MoE often RCEIL)
- iters / method (RTN=0 / tuning=200 / model_free) vs model scale
- Layers to KEEP FP16 with per-item rationale: lm_head, embedding, router/gate,
  self_attn (MLA or FP4 case), shared_experts, vision_tower/mm_projector, linear_attn
- layer_config mixed precision (e.g. experts->MXFP4, sensitive layers->MXFP8)
- Quantization target confirmed: routed experts MUST be quantized (bulk of params)
- Operator-fixed config review: for EACH hard constraint, confirm the command
  honors it and state whether you agree; if you'd recommend a different value,
  say so here as advice (the command still obeys the operator).

## 6. Recommended command (auto_round CLI mode)
```autoround-cmd
auto-round --model_name <model> --scheme ... --output_dir <REQUIRED> --format ... ...
```

## 7. Accuracy expectation & validation
- Expected retention, validation method (lm-eval vs BF16 baseline)

## 8. Optional variants & caveats
- More aggressive / more conservative options; known pitfalls (ignore-name vs actual
  weight-name mismatch causing load errors, etc.)
- If any operator-fixed value looks suboptimal for this model, record your
  recommended alternative here (advisory only — the command above still obeys it).
"""

# Concrete auto_round CLI exemplars (tuning / model_free / mixed-precision).  Always
# pass the model via ``--model_name`` (this auto-round build mis-parses positional).
CLI_EXAMPLES = """### Example A -- MoE mixed precision + tuning (experts MXFP4, rest MXFP8)
```bash
auto-round --model_name <MODEL> --iters 200 --scheme MXFP8 \\
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,multi_modal_projector \\
  --layer_config "{mlp.experts:{bits:4,data_type:mx_fp}}" \\
  --format llm_compressor --output_dir <OUT> --device_map auto
```

### Example B -- huge MoE, pure RTN (iters=0), MXFP4_RCEIL
```bash
auto-round --model_name <MODEL> --scheme MXFP4_RCEIL --iters 0 \\
  --fp_layers lm_head,mlp.gate,self_attn \\
  --format llm_compressor --output_dir <OUT> --device_map auto
```

### Example C -- model_free (no model load, cheapest; fp8 source or NON-pre-packed huge weights)
# NOTE: model_free only dequantizes fp8 sources. NEVER use --model_free when the
# MoE experts are already MXFP4-packed (weights named `*_blocks`/`*_scales`, e.g.
# gpt-oss): it cannot dequant that native layout and emits a broken checkpoint.
# For such models use the model-load path (no --model_free) + --iters 0 (RTN).
```bash
auto-round --model_name <MODEL> --model_free \\
  --scheme W4A16 --ignore_layers lm_head,embed_tokens,mlp.gate \\
  --format auto_round --output_dir <OUT>
```

### Example D -- dense small model, tuning(iters=200)
```bash
auto-round --model_name <MODEL> --scheme MXFP4 --iters 200 \\
  --fp_layers lm_head,self_attn \\
  --format llm_compressor --output_dir <OUT> --device_map auto
```
"""

# Diffusion (DiT) CLI exemplars, used when the model is a diffusion pipeline
# (``ModelAnalysis.is_diffusion``). Key differences from the LLM examples:
# ``--format auto_round`` is mandatory for vllm-omni deployment, calibration
# defaults to coco2014, and ``--num_inference_steps/--guidance_scale`` follow
# the model's inference convention (distilled models: few steps, no CFG).
DIT_CLI_EXAMPLES = """### Example DiT-A -- distilled DiT (Z-Image-Turbo / FLUX.2-klein), W4A16 RTN
```bash
auto-round --model_name <LOCAL_PATH> --scheme W4A16 --iters 0 --disable_opt_rtn \\
  --disable_model_free --dataset coco2014 --num_inference_steps 8 --guidance_scale 0.0 \\
  --format auto_round --output_dir <OUT>
```

### Example DiT-B -- FLUX.1-dev, W4A16 with tuning (calibration cost is real)
```bash
auto-round --model_name <LOCAL_PATH> --scheme W4A16 --iters 200 --nsamples 128 \\
  --dataset coco2014 --num_inference_steps 28 --guidance_scale 3.5 \\
  --format auto_round --output_dir <OUT>
```
# NOTE: iters>0 runs the FULL pipeline nsamples x num_inference_steps times for
# calibration. auto-round ships dual-output block registrations only for FLUX.1
# (FluxTransformerBlock / FluxSingleTransformerBlock); other architectures (e.g.
# Flux2TransformerBlock) need register_diffusion_output() first -- verify before
# enabling tuning there. Pure RTN (iters 0) skips calibration entirely and is
# always the safe first run.
# NOTE 2: <LOCAL_PATH> not a hub id -- the regular diffusion loader opens
# <path>/model_index.json directly (no hub resolution); download first. And
# ALWAYS pass --disable_model_free with --iters 0 --disable_opt_rtn: the
# model-free auto-route silently quantizes 0 layers for diffusion models.

### Example DiT-C -- MXFP8 RTN (vllm-omni supports MXFP8 since PR #3782)
```bash
auto-round --model_name <LOCAL_PATH> --scheme MXFP8 --iters 0 --disable_opt_rtn \\
  --disable_model_free --dataset coco2014 --num_inference_steps 8 --guidance_scale 0.0 \\
  --format auto_round --output_dir <OUT>
```
"""

MOE_CONFIG_KEYS = (
    "num_experts_per_tok", "n_routed_experts", "num_local_experts",
    "num_experts", "moe_num_experts",
)

# Bare weight-leaf names that would substring-match routed-expert tensors; protecting
# these into FP16 would skip the quantization target.  Qualified forms (``mlp.down_proj``)
# are safe -- they never substring-match ``mlp.experts.N.down_proj``.
_BARE_EXPERT_LEAVES = {
    "gate_proj", "up_proj", "down_proj", "gate_up_proj", "w1", "w2", "w3", "experts",
}


@dataclass
class ModelAnalysis:
    model_id: str
    ok: bool = True
    error: str | None = None
    architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    is_moe: bool = False
    num_experts: int | None = None
    num_layers: int | None = None
    has_router: bool = False
    has_shared_experts: bool = False
    has_self_attn: bool = False
    has_vision: bool = False
    has_mla: bool = False
    is_diffusion: bool = False
    router_names: list[str] = field(default_factory=list)
    sample_modules: list[str] = field(default_factory=list)


def _normalize(name: str) -> str:
    return _TRAIL_IDX_RE.sub(".N", _LAYER_IDX_RE.sub(".N.", name))


def _rel(module_key: str) -> str:
    k = module_key
    for suf in (".weight", ".bias"):
        k = k.removesuffix(suf)
    parts = [p for p in k.split(".") if p not in ("model", "transformer")]
    return ".".join(parts[-3:])


def _hf_config(model_id: str, revision: str, token: str | None,
               filename: str = "config.json") -> dict:
    if Path(model_id).is_dir():
        return json.loads((Path(model_id) / filename).read_text())
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model_id, filename, revision=revision, token=token)
    return json.loads(Path(path).read_text())


def _has_model_index(model_id: str, revision: str, token: str | None) -> bool:
    """True when the repo carries a diffusers ``model_index.json`` (diffusion pipeline)."""
    if Path(model_id).is_dir():
        return (Path(model_id) / "model_index.json").is_file()
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub.utils import EntryNotFoundError
    except Exception:  # noqa: BLE001
        EntryNotFoundError = Exception  # type: ignore[assignment,misc]
    try:
        hf_hub_download(model_id, "model_index.json", revision=revision, token=token)
        return True
    except EntryNotFoundError:
        return False
    except Exception as exc:  # noqa: BLE001
        _log.debug("model_index.json probe failed: %s", exc)
        return False


def _tensor_names(model_id: str, revision: str, token: str | None, *,
                  index_file: str = "model.safetensors.index.json",
                  single_file: str = "model.safetensors") -> list[str]:
    """Tensor names from the safetensors index (no weights); fall back to a header read."""
    if Path(model_id).is_dir():
        idx = Path(model_id) / index_file
        if idx.is_file():
            return sorted(json.loads(idx.read_text()).get("weight_map", {}).keys())
        names: list[str] = []
        for shard in sorted(Path(model_id).glob("*.safetensors")):
            try:
                with open(shard, "rb") as fh:
                    (n,) = struct.unpack("<Q", fh.read(8))
                    header = json.loads(fh.read(n))
                names.extend(k for k in header if k != "__metadata__")
            except Exception:  # noqa: BLE001 - skip unreadable shard
                continue
        return sorted(names)
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub.utils import EntryNotFoundError
    except Exception:  # noqa: BLE001
        EntryNotFoundError = Exception  # type: ignore[assignment,misc]
    try:
        idx = hf_hub_download(model_id, index_file, revision=revision, token=token)
        return sorted(json.loads(Path(idx).read_text()).get("weight_map", {}).keys())
    except EntryNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        _log.debug("index read failed, trying header: %s", exc)
    try:
        path = hf_hub_download(model_id, single_file, revision=revision, token=token)
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        return sorted(k for k in header if k != "__metadata__")
    except Exception:  # noqa: BLE001
        return []


def analyze_model(model_id: str, *, revision: str = "main",
                  token: str | None = None) -> ModelAnalysis:
    """Download-free structural analysis of a HF model (config + index names only)."""
    res = ModelAnalysis(model_id=model_id)
    index_file, single_file = "model.safetensors.index.json", "model.safetensors"
    try:
        cfg = _hf_config(model_id, revision, token)
    except Exception as exc:  # noqa: BLE001
        # Diffusion pipelines carry no root config.json; the quantization target
        # (transformer/unet) has its own config in a subfolder.
        if not _has_model_index(model_id, revision, token):
            res.ok = False
            res.error = f"config.json unavailable: {exc}"
            return res
        res.is_diffusion = True
        try:
            cfg = _hf_config(model_id, revision, token,
                             filename="transformer/config.json")
        except Exception as exc2:  # noqa: BLE001
            res.ok = False
            res.error = f"transformer/config.json unavailable: {exc2}"
            return res
        index_file = "transformer/diffusion_pytorch_model.safetensors.index.json"
        single_file = "transformer/diffusion_pytorch_model.safetensors"

    res.architectures = list(cfg.get("architectures", []) or [])
    res.model_type = cfg.get("model_type", "") or ""
    for k in MOE_CONFIG_KEYS:
        v = cfg.get(k)
        if isinstance(v, int) and v > 1:
            res.is_moe = True
            res.num_experts = res.num_experts or v
    arch_str = " ".join(res.architectures).lower()
    if not res.is_moe and any(t in arch_str for t in ("moe", "sparse", "mixtral", "grok")):
        res.is_moe = True
    res.num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    if any(cfg.get(k) for k in ("kv_lora_rank", "q_lora_rank", "qk_rope_head_dim")):
        res.has_mla = True

    norm = sorted({_normalize(n) for n in _tensor_names(
        model_id, revision, token, index_file=index_file, single_file=single_file)})
    res.sample_modules = norm[:60]
    routers: set[str] = set()
    for m in norm:
        low = m.lower()
        if (m.endswith((".gate.weight", ".router.gate.weight")) or ".router" in low
                or m.split(".")[-1] == "gate") and "gate_proj" not in low:
            res.has_router = True
            routers.add(_rel(m))
        if "shared_expert" in low:
            res.has_shared_experts = True
        if "self_attn" in low or ".attn." in low:
            res.has_self_attn = True
        if any(v in low for v in ("vision", "visual", "multi_modal", "mm_projector",
                                  "image_", "vit")):
            res.has_vision = True
    res.router_names = sorted(routers)[:12]
    return res


_RECIPE_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "recipes" / "auto_round_recipe.md"
_DIFFUSION_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "recipes" / "auto_round_diffusion.md"


def load_recipe_knowledge(*, include_diffusion: bool = False) -> str:
    """Return the packaged distilled auto_round recipe knowledge (best-effort).

    This is the reference corpus that grounds the agent's structured analysis in
    real-world MXFP4/MoE recipes rather than guesswork.  Missing/unreadable asset
    degrades gracefully to an empty string (the flow still works, just less informed).
    ``include_diffusion`` appends the distilled diffusion/VLM knowledge; pass it for
    diffusion pipelines (``ModelAnalysis.is_diffusion``) so DiT-specific constraints
    (transformer-only quantization, coco2014 calibration, auto_round export for
    vllm-omni) reach the prompt without bloating LLM-only runs.
    """
    try:
        text = _RECIPE_KNOWLEDGE_PATH.read_text(encoding="utf-8")
    except OSError:
        text = ""
    if include_diffusion:
        try:
            extra = _DIFFUSION_KNOWLEDGE_PATH.read_text(encoding="utf-8")
        except OSError:
            extra = ""
        if extra:
            text = f"{text}\n\n{extra}" if text else extra
    return text


def build_optimize_prompt(a: ModelAnalysis, *, scheme: str, method: str,
                          export_format: str, output_dir: str,
                          skill_recipe: str = "", user_context: str = "",
                          recipe_knowledge: str = "",
                          constraints: list[QuantConstraint] | None = None) -> str:
    """Ask the agent for a structured quantization recipe + one optimal command."""
    facts = [
        f"model: {a.model_id}",
        f"architectures: {a.architectures}",
        f"model_type: {a.model_type}",
        f"is_moe: {a.is_moe} (num_experts={a.num_experts})",
        f"num_layers: {a.num_layers}",
        f"has_router: {a.has_router} routers={a.router_names}",
        f"has_shared_experts: {a.has_shared_experts}",
        f"has_mla: {a.has_mla}  has_vision: {a.has_vision}",
        f"is_diffusion: {a.is_diffusion}",
    ]
    kind = "diffusion (DiT) pipeline" if a.is_diffusion else "LLM"
    intro = (
        "You are a senior model-quantization engineer.  Produce a **quantization "
        f"recipe document** (Markdown) for the {kind} `{a.model_id}` at scheme {scheme} "
        f"(method {method}): analyse the structure, justify each decision, and end "
        "with ONE ready-to-run `auto-round` CLI command."
    )
    if a.is_diffusion:
        intro += (
            "  Only the transformer/unet is quantized (auto-round handles this "
            "automatically); text encoders and the VAE stay untouched."
        )
    parts = [
        intro,
        "## Model structure (download-free analysis)\n" + "\n".join(facts),
        "## Sample module names (normalized)\n" + "\n".join(a.sample_modules[:40]),
    ]
    if constraints:
        directives = "\n".join(f"- {c.label}" for c in constraints)
        parts.append(
            "## Operator-fixed configuration (HARD CONSTRAINTS — obey verbatim)\n"
            "These come from the user's run config and OVERRIDE your own analysis, the "
            "recipe knowledge, and the CLI examples. Your final command MUST satisfy "
            "every one of them:\n" + directives + "\n\n"
            "If you judge any fixed value suboptimal or risky, STILL honor it in the "
            "command, but record your understanding + recommended alternative in "
            "sections 5 and 8 of the recipe document (advisory only — do not change "
            "the command to disagree)."
        )
    if recipe_knowledge:
        parts.append("## auto_round recipe knowledge (MUST consult)\n" + recipe_knowledge)
    if skill_recipe:
        parts.append("## Recipe knowledge (auto_quant skill)\n" + skill_recipe)
    if user_context:
        parts.append("## Operator context (HIGH PRIORITY)\n" + user_context)
    parts.append(
        "## auto-round diffusion CLI examples\n" + DIT_CLI_EXAMPLES
        if a.is_diffusion else
        "## auto-round CLI examples\n" + CLI_EXAMPLES
    )
    parts.append("## Recipe document structure (follow this exactly)\n" + RECIPE_DOC_TEMPLATE)
    hard = (
        "## Hard requirements for your command\n"
        f"- MUST include `--output_dir {output_dir}` exactly.\n"
        f"- MUST include `--format {export_format}` and `--model_name {a.model_id}`.\n"
        "- For MoE models: FP-protect the ROUTER/GATE only (never the routed experts).\n"
        "  Only put experts back at low bits via an explicit `--layer_config`.\n"
        "- Do not invent flags; use only real auto-round options.\n"
        "- Prefer the simplest command that is correct.\n"
    )
    if a.is_diffusion:
        hard += (
            "- Diffusion: `--format auto_round` is mandatory for vllm-omni deployment; "
            "MXFP4 checkpoints are loadable on vllm-omni main (PR #5544 merged "
            "2026-08-25; use main or a release carrying it, older installs reject "
            "MXFP4); calibration runs the full pipeline per sample x step, "
            "so start at `--iters 0` unless dual-output block registration for this "
            "architecture is confirmed in auto-round.\n"
        )
    parts.append(
        hard + "\n## Output format (strict)\n"
        "- The whole output IS the Markdown recipe document (no chit-chat).\n"
        "- In section 6, give the final command inside **exactly one** ```autoround-cmd\n"
        "  fenced block, on a single line, starting with `auto-round`."
    )
    return "\n\n".join(parts)


def extract_command(log_text: str) -> str | None:
    """Pull the last ``auto-round`` command from the agent's output (code block first)."""
    candidates: list[str] = []
    for m in _CMD_BLOCK_RE.finditer(log_text or ""):
        body = re.sub(r"\\\s*\n\s*", " ", m.group(1).strip())
        for line in body.splitlines():
            line = line.strip().lstrip("$ ").strip()
            if line.startswith(("auto-round", "auto_round")):
                candidates.append(line)
                break
    if not candidates:
        joined = re.sub(r"\\\s*\n\s*", " ", log_text or "")
        for line in joined.splitlines():
            s = line.strip().lstrip("$ ").strip()
            if s.startswith(("auto-round", "auto_round")):
                candidates.append(s)
    return candidates[-1] if candidates else None


def _opt(toks: list[str], name: str) -> str | None:
    for i, t in enumerate(toks):
        if t == name and i + 1 < len(toks):
            return toks[i + 1]
        if t.startswith(name + "="):
            return t.split("=", 1)[1]
    return None


def ensure_model_name(cmd: str, model: str) -> str:
    """Guarantee the command carries an explicit ``--model_name <value>``."""
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return cmd
    if not toks:
        return cmd
    head, rest = toks[0], toks[1:]
    flags = ("--model_name", "--model", "--model_name_or_path")
    model_val: str | None = None
    out: list[str] = []
    i = 0
    while i < len(rest):
        t = rest[i]
        if t in flags:
            nxt = rest[i + 1] if i + 1 < len(rest) else None
            if nxt is not None and not nxt.startswith("-"):
                model_val = model_val or nxt
                i += 2
                continue
            i += 1
            continue
        if "=" in t and t.split("=", 1)[0] in flags:
            v = t.split("=", 1)[1]
            if v and not v.startswith("-"):
                model_val = model_val or v
            i += 1
            continue
        out.append(t)
        i += 1
    if model_val is None:
        if model in out:
            out.remove(model)
            model_val = model
        elif out and not out[0].startswith("-"):
            model_val = out.pop(0)
        else:
            model_val = model
    out = [t for t in out if t != model_val]
    return " ".join(shlex.quote(t) for t in [head, "--model_name", model_val, *out])


def _protects_experts(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    if t in _BARE_EXPERT_LEAVES:
        return True
    return "expert" in t.lower()


@dataclass(frozen=True)
class QuantConstraint:
    """One operator-fixed recipe constraint the agent's command MUST satisfy.

    Derived from the fields the user *explicitly set* in ``quant:`` (pydantic
    ``model_fields_set``) so agent-optimize augments rather than overrides the
    operator's intent. ``kind`` drives both the prompt directive and validation:

    * ``value``   -- ``--flag`` must equal ``value`` (case-insensitive).
    * ``present`` -- boolean ``--flag`` must be present.
    * ``absent``  -- ``--flag`` must NOT appear (e.g. forbid ``--model_free``).
    * ``subset``  -- every comma token in ``value`` must appear in the flag's list.
    * ``haskeys`` -- the flag's value must contain each comma-listed key.
    """

    label: str          # human directive rendered into the prompt
    flag: str           # auto-round flag it maps to
    kind: str           # value | present | absent | subset | haskeys
    value: str = ""     # expected value / comma tokens / keys (per ``kind``)

    def satisfied_by(self, toks: list[str]) -> bool:
        present = self.flag in toks or any(t.startswith(self.flag + "=") for t in toks)
        if self.kind == "present":
            return present
        if self.kind == "absent":
            return not present
        got = _opt(toks, self.flag)
        if self.kind == "value":
            return got is not None and got.strip().lower() == self.value.strip().lower()
        if self.kind == "subset":
            if got is None:
                return False
            have = {x.strip() for x in got.split(",") if x.strip()}
            want = {x.strip() for x in self.value.split(",") if x.strip()}
            return want.issubset(have)
        if self.kind == "haskeys":
            return got is not None and all(k in got for k in self.value.split(",") if k)
        return True


def build_quant_constraints(quant_cfg: QuantConfig) -> list[QuantConstraint]:
    """Translate the user's *explicitly set* quant fields into hard constraints.

    Only fields present in ``model_fields_set`` are enforced, so schema defaults
    never override the agent's structural analysis -- but anything the operator
    actually typed takes precedence over the agent's own recommendation.
    """
    setf = getattr(quant_cfg, "model_fields_set", set())
    cons: list[QuantConstraint] = []

    if "scheme" in setf:
        cons.append(QuantConstraint(
            f"scheme MUST be `{quant_cfg.scheme}`", "--scheme", "value", quant_cfg.scheme))
    if "method" in setf:
        if quant_cfg.method == "model_free":
            cons.append(QuantConstraint(
                "MUST use the calibration-free model_free path (pass `--model_free`)",
                "--model_free", "present"))
        else:
            cons.append(QuantConstraint(
                "MUST use the model-load path -- do NOT pass `--model_free`",
                "--model_free", "absent"))
    if "export_format" in setf:
        cons.append(QuantConstraint(
            f"export/--format MUST be `{quant_cfg.export_format}`",
            "--format", "value", quant_cfg.export_format))
    if "iters" in setf and quant_cfg.iters is not None:
        cons.append(QuantConstraint(
            f"MUST pass `--iters {quant_cfg.iters}`", "--iters", "value", str(quant_cfg.iters)))
    if "nsamples" in setf and quant_cfg.nsamples is not None:
        cons.append(QuantConstraint(
            f"MUST pass `--nsamples {quant_cfg.nsamples}`",
            "--nsamples", "value", str(quant_cfg.nsamples)))
    if "ignore_layers" in setf and quant_cfg.ignore_layers:
        cons.append(QuantConstraint(
            f"`--ignore_layers` MUST include: {quant_cfg.ignore_layers}",
            "--ignore_layers", "subset", quant_cfg.ignore_layers))
    if "layer_config" in setf and quant_cfg.layer_config:
        keys = ",".join(str(k) for k in quant_cfg.layer_config)
        cons.append(QuantConstraint(
            f"`--layer_config` MUST cover keys: {keys}", "--layer_config", "haskeys", keys))
    if "algorithm" in setf and quant_cfg.algorithm:
        cons.append(QuantConstraint(
            f"`--algorithm` MUST be `{quant_cfg.algorithm}`",
            "--algorithm", "value", quant_cfg.algorithm))
    if "enable_alg_ext" in setf and quant_cfg.enable_alg_ext:
        cons.append(QuantConstraint(
            "MUST pass `--enable_alg_ext`", "--enable_alg_ext", "present"))
    if "hadamard_type" in setf and quant_cfg.hadamard_type:
        cons.append(QuantConstraint(
            f"`--rotation-hadamard-type` MUST be `{quant_cfg.hadamard_type}`",
            "--rotation-hadamard-type", "value", quant_cfg.hadamard_type))
    if "opt_rtn" in setf and quant_cfg.opt_rtn is not None:
        flag = "--enable_opt_rtn" if quant_cfg.opt_rtn else "--disable_opt_rtn"
        cons.append(QuantConstraint(f"MUST pass `{flag}`", flag, "present"))
    if "quant_lm_head" in setf and quant_cfg.quant_lm_head:
        cons.append(QuantConstraint(
            "MUST pass `--quant_lm_head`", "--quant_lm_head", "present"))
    if "static_kv_fp8" in setf and quant_cfg.static_kv_fp8:
        cons.append(QuantConstraint(
            "MUST pass `--static_kv_dtype fp8`", "--static_kv_dtype", "value", "fp8"))
    if "static_attention_fp8" in setf and quant_cfg.static_attention_fp8:
        cons.append(QuantConstraint(
            "MUST pass `--static_attention_dtype fp8`",
            "--static_attention_dtype", "value", "fp8"))
    return cons


def validate_command(cmd: str, a: ModelAnalysis, output_dir: str,
                     constraints: list[QuantConstraint] | None = None) -> tuple[bool, str]:
    """Return ``(ok, reason)``; enforce operator-fixed config + expert-safety."""
    try:
        toks = shlex.split(cmd)
    except ValueError as exc:
        return False, f"unparseable command: {exc}"
    if not toks or toks[0] not in ("auto-round", "auto_round"):
        return False, "command must start with auto-round"
    if _opt(toks, "--output_dir") != output_dir:
        return False, f"--output_dir must be exactly {output_dir}"
    if not (_opt(toks, "--model_name") or _opt(toks, "--model")
            or (len(toks) >= 2 and not toks[1].startswith("-"))):
        return False, "missing model (positional path or --model_name)"

    # Operator-fixed config wins: reject any command that violates an explicit
    # user quant setting (so agent-optimize augments, never overrides, intent).
    for con in constraints or ():
        if not con.satisfied_by(toks):
            return False, f"violates operator-fixed config: {con.label}"

    layer_cfg = _opt(toks, "--layer_config") or ""
    experts_reenabled = "expert" in layer_cfg.lower()
    protected = [
        x.strip()
        for x in ((_opt(toks, "--fp_layers") or "") + "," +
                  (_opt(toks, "--ignore_layers") or "")).split(",")
        if x.strip()
    ]
    if a.is_moe and not experts_reenabled:
        nuking = [t for t in protected if _protects_experts(t)]
        if nuking:
            return False, (f"would fp-protect routed experts {nuking} without a "
                           "--layer_config experts override")
        if protected and not any(("gate" in t or "router" in t) for t in protected):
            return False, "MoE run must fp-protect the router/gate"
    return True, "ok"


# Agent runner signature: (tier, prompt, log_path, session_id) -> output text.
OptimizeAgentRunner = Callable[[Tier, str, Path, str], str]

_VERDICT_TAIL_RE = re.compile(
    r"\n(VERDICT|FIXABLE|UNFIXABLE|ERROR_CLASS|ROOT_CAUSE)\b.*$", re.DOTALL)


def save_recipe_doc(log_text: str, path: Path, model_id: str) -> None:
    """Persist the agent's Markdown recipe document (trims chatter before the H1)."""
    text = log_text or ""
    if text.lstrip().startswith("# "):
        i = text.find("# ")
    else:
        i = text.find("\n# ")
    if i != -1:
        text = text[i:]
    text = _VERDICT_TAIL_RE.sub("\n", text)
    header = (f"<!-- Auto-generated quantization recipe for {model_id} "
              f"by aqa agent-optimize. -->\n\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + text.strip() + "\n", encoding="utf-8")


def run_agent_optimize(
    model: str,
    *,
    scheme: str,
    method: str,
    export_format: str,
    output_dir: str,
    tier: Tier,
    run_agent: OptimizeAgentRunner,
    revision: str = "main",
    token: str | None = None,
    skill_recipe: str = "",
    user_context: str = "",
    recipe_knowledge: str | None = None,
    constraints: list[QuantConstraint] | None = None,
    log_dir: Path | None = None,
) -> str | None:
    """Analyse the model, ask the agent for a command, validate it; None on any miss."""
    a = analyze_model(model, revision=revision, token=token)
    if not a.ok:
        _log.warning("agent-optimize: analysis failed: %s", a.error)
        return None
    if recipe_knowledge is None:
        recipe_knowledge = load_recipe_knowledge(include_diffusion=a.is_diffusion)
    prompt = build_optimize_prompt(
        a, scheme=scheme, method=method, export_format=export_format,
        output_dir=output_dir, skill_recipe=skill_recipe, user_context=user_context,
        recipe_knowledge=recipe_knowledge, constraints=constraints,
    )
    log_dir = Path(log_dir or Path.cwd())
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "optimize.log"
    output = run_agent(tier, prompt, log_path, f"optimize_{model.split('/')[-1]}")
    # A shared agent runner may hand back a structured outcome (fix-loop's
    # AgentOutcome) rather than a bare str; we only need its text for extraction.
    output = getattr(output, "text", output)

    # Persist the human-readable recipe doc regardless of command validity -- the
    # structured analysis is valuable for auditing even when the command is rejected.
    try:
        save_recipe_doc(output, log_dir / "quantization_recipe.md", model)
        _log.info("agent-optimize: recipe doc -> %s", log_dir / "quantization_recipe.md")
    except OSError as exc:
        _log.warning("agent-optimize: could not save recipe doc (%s)", exc)

    cmd = extract_command(output)
    if not cmd:
        _log.warning("agent-optimize: no auto-round command produced")
        return None
    cmd = ensure_model_name(cmd, model)
    ok, reason = validate_command(cmd, a, output_dir, constraints)
    if not ok:
        _log.warning("agent-optimize: rejected command (%s)", reason)
        return None
    _log.info("agent-optimize: validated command")
    return cmd
