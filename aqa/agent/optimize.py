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

# The recipe-document template and CLI exemplars live in the single canonical
# domain document (``recipes/auto_round_recipe.md`` §5/§6/§7).  The prompt below
# only adds per-run facts + operator constraints; it must never re-embed a second
# copy of the domain rules (that was the source-drift bug this refactor closes).

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


def _hf_config(model_id: str, revision: str, token: str | None) -> dict:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model_id, "config.json", revision=revision, token=token)
    return json.loads(Path(path).read_text())


def _tensor_names(model_id: str, revision: str, token: str | None) -> list[str]:
    """Tensor names from the safetensors index (no weights); fall back to a header read."""
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub.utils import EntryNotFoundError
    except Exception:  # noqa: BLE001
        EntryNotFoundError = Exception  # type: ignore[assignment,misc]
    try:
        idx = hf_hub_download(model_id, "model.safetensors.index.json",
                              revision=revision, token=token)
        return sorted(json.loads(Path(idx).read_text()).get("weight_map", {}).keys())
    except EntryNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        _log.debug("index read failed, trying header: %s", exc)
    try:
        path = hf_hub_download(model_id, "model.safetensors", revision=revision, token=token)
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
    try:
        cfg = _hf_config(model_id, revision, token)
    except Exception as exc:  # noqa: BLE001
        res.ok = False
        res.error = f"config.json unavailable: {exc}"
        return res

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

    norm = sorted({_normalize(n) for n in _tensor_names(model_id, revision, token)})
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


def load_recipe_knowledge(knowledge_dir: str | Path | None = None,
                          *, use_synced: bool = True) -> str:
    """Return the canonical auto_round recipe knowledge (single source).

    Prefers the synced ``references/auto_round_recipe.md`` copy when a
    ``knowledge_dir`` is supplied and ``use_synced`` is true; otherwise falls
    back to the packaged document.  Missing/unreadable assets degrade to an
    empty string (the flow still works, just less informed).
    """
    if use_synced and knowledge_dir is not None:
        synced = Path(knowledge_dir) / "references" / "auto_round_recipe.md"
        try:
            if synced.is_file():
                return synced.read_text(encoding="utf-8")
        except OSError:
            pass
    try:
        return _RECIPE_KNOWLEDGE_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


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
    ]
    parts = [
        ("You are a senior LLM-quantization engineer.  Produce a **quantization "
         f"recipe document** (Markdown) for `{a.model_id}` at scheme {scheme} "
         f"(method {method}): analyse the structure, justify each decision, and end "
         "with ONE ready-to-run `auto-round` CLI command."),
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
        "## Recipe document template + CLI exemplars\n"
        "They are §5/§6 of the recipe knowledge above; follow that structure and "
        "those exemplars exactly.  The knowledge document is the single source — "
        "do not expect a second copy in this prompt."
    )
    parts.append(
        "## Hard requirements for your command\n"
        f"- MUST include `--output_dir {output_dir}` exactly.\n"
        f"- MUST include `--format {export_format}` and `--model_name {a.model_id}`.\n"
        "- Obey the safety rules in §7 of the recipe knowledge (routed experts "
        "must be quantized; never FP-protect them without an explicit "
        "`--layer_config` override).\n"
        "- Do not invent flags; use only real auto-round options.\n"
        "- Prefer the simplest command that is correct.\n\n"
        "## Output format (strict)\n"
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
    knowledge_dir: str | Path | None = None,
    use_synced: bool = True,
) -> str | None:
    """Analyse the model, ask the agent for a command, validate it; None on any miss."""
    a = analyze_model(model, revision=revision, token=token)
    if not a.ok:
        _log.warning("agent-optimize: analysis failed: %s", a.error)
        return None
    if recipe_knowledge is None:
        recipe_knowledge = load_recipe_knowledge(
            knowledge_dir=knowledge_dir, use_synced=use_synced)
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
