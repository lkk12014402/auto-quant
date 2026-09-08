"""Deterministic multi-recipe candidate generation for ``aqa sweep``.

When the operator does not pin an exact quant config, the sweep explores a
small set of candidate plans derived from the model's *structure* (the same
download-free analysis the optimize agent uses) instead of guessing one:

- ``conservative`` — experts-only (MoE) / linears-only (dense) at MXFP4 with
  every sensitive module family kept BF16.  This is the lower-bound run: if it
  loses accuracy, the plan itself is wrong, not the scheme.
- ``aggressive``  — same scheme but a wider quantization net (shared expert and
  dense MLPs included; only router/embed/indexer-style families stay BF16).
- ``mixed``       — MoE only: MXFP8 everywhere eligible, routed experts pulled
  down to MXFP4 via ``layer_config`` (mxfp8/mxfp4 混合精度).
- ``tuning``      — the conservative layer plan through the calibrated
  autoround path (``iters``), for when RTN's drop is unacceptable.

Every candidate is a plain ``QuantConfig`` patch + a human-readable rationale,
so the owner can review the plan before any GPU hour is spent (``--dry-run``),
and every entry is overridable from yaml (``quant.recipe_sweep_candidates``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agent.optimize import ModelAnalysis
from ..config import QuantConfig
from ..logging_util import get_logger

_log = get_logger("quantize.candidates")

# Sensitive module families -> ignore_layers keywords.  Keywords are matched as
# substrings of the full tensor name by auto-round, so one word covers a whole
# family (``self_attn`` covers q/k/v/o + indexer).
_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "self_attn": ("self_attn", ".attn."),
    "linear_attn": ("linear_attn",),
    "shared_expert": ("shared_expert",),
    "hyper_connection": ("hyper_connection", "hc_"),
    "ple": ("ple",),
    "mtp": ("mtp",),
    "visual": ("visual", "vision"),
    "indexer": ("indexer",),
    "eh_proj": ("eh_proj",),
}

# Families that even the aggressive plan never quantizes: routing, embeddings
# and sparse-attention selectors are precision-critical (and most are
# hard-skipped by auto-round anyway — listing them is harmless and explicit).
_ALWAYS_KEEP = ("lm_head", "embed_tokens", "mlp.gate", "indexer", "eh_proj")


@dataclass
class Candidate:
    """One sweep entry: a QuantConfig patch plus the why."""

    name: str
    overrides: dict[str, Any]
    rationale: str = ""
    warnings: list[str] = field(default_factory=list)
    # Per-candidate eval tweak, merged into eval.lm_eval.model_args for this
    # candidate's run only — e.g. FP8-KV variants must evaluate with
    # ``kv_cache_dtype=fp8`` while the plain variant must not.
    eval_model_args: dict[str, Any] = field(default_factory=dict)


def detect_module_families(tensor_names: list[str]) -> list[str]:
    """Which sensitive families actually occur in the checkpoint's tensor names."""
    found: list[str] = []
    for family, needles in _FAMILY_KEYWORDS.items():
        if any(needle in name for name in tensor_names for needle in needles):
            found.append(family)
    return found


def _ignore_for(analysis: ModelAnalysis, families: list[str], *,
                drop: tuple[str, ...] = ()) -> str:
    """Build the ignore_layers string for a candidate."""
    keep = [f for f in families if f not in drop]
    kw = [f for f in keep]
    if analysis.has_vision and "visual" not in kw:
        kw.append("visual")
    return ",".join(dict.fromkeys(kw))  # dedupe, stable order


def builtin_candidates(analysis: ModelAnalysis, *,
                       base_scheme: str = "MXFP4",
                       tuning_iters: int = 200) -> list[Candidate]:
    """The default candidate set for a model, given its structure."""
    families = detect_module_families(analysis.sample_modules)
    cands: list[Candidate] = []

    conservative_ignore = _ignore_for(analysis, families)
    cands.append(Candidate(
        name="conservative",
        overrides={
            "scheme": base_scheme, "method": "model_free",
            "ignore_layers": conservative_ignore,
        },
        rationale=(f"下限基线：{'路由专家' if analysis.is_moe else 'Linear'}独享 "
                   f"{base_scheme}，敏感模块族（{conservative_ignore or '无'}）全部保 BF16"),
    ))

    aggressive_ignore = _ignore_for(analysis, families, drop=("shared_expert",))
    cands.append(Candidate(
        name="aggressive",
        overrides={
            "scheme": base_scheme, "method": "model_free",
            "ignore_layers": ",".join(
                dict.fromkeys([*filter(None, aggressive_ignore.split(",")),
                               *[k for k in _ALWAYS_KEEP
                                 if k not in aggressive_ignore]])),
        },
        rationale=("激进：量化网放开到 shared_expert/稠密 MLP，仅 router/embedding/"
                   "indexer 类保 BF16（掉点风险更高，换取体积收益）"),
    ))

    if analysis.is_moe:
        mixed_ignore = _ignore_for(analysis, families, drop=("shared_expert",))
        cands.append(Candidate(
            name="mixed",
            overrides={
                "scheme": "MXFP8", "method": "model_free",
                "ignore_layers": mixed_ignore,
                "layer_config": {"mlp.experts": {"bits": 4, "data_type": "mx_fp"}},
            },
            rationale=("混合：可量化 Linear 走 MXFP8，路由专家经 layer_config 压到 "
                       "MXFP4（moe=mxfp4、其它=mxfp8 的折中档）"),
        ))
    else:
        cands.append(Candidate(
            name="mixed",
            overrides={"scheme": "MXFP8", "method": "model_free",
                       "ignore_layers": conservative_ignore},
            rationale="混合（dense 退化档）：全量 MXFP8，敏感族保 BF16",
        ))

    cands.append(Candidate(
        name="tuning",
        overrides={
            "scheme": base_scheme, "method": "autoround", "iters": tuning_iters,
            "ignore_layers": conservative_ignore,
        },
        rationale=(f"调优：保守层计划走校准路径（iters={tuning_iters}），"
                   "RTN 掉点不可接受时的精度上限档（耗时最长）"),
    ))
    return cands


def custom_candidates(items: list[dict[str, Any]]) -> list[Candidate]:
    """User-defined candidates from ``quant.recipe_sweep_candidates``."""
    fields = set(QuantConfig.model_fields)
    out: list[Candidate] = []
    for i, item in enumerate(items):
        item = dict(item)
        name = str(item.pop("name", f"candidate{i}"))
        rationale = str(item.pop("description", ""))
        eval_args = dict(item.pop("eval_model_args", {}) or {})
        unknown = set(item) - fields
        if unknown:
            raise ValueError(
                f"recipe_sweep_candidates[{i}] ({name}) has unknown QuantConfig "
                f"fields: {sorted(unknown)}")
        out.append(Candidate(name=name, overrides=item, rationale=rationale,
                             eval_model_args=eval_args))
    return out


def resolve_candidates(analysis: ModelAnalysis,
                       quant_cfg: QuantConfig) -> list[Candidate]:
    """User-provided list wins; otherwise the built-in set."""
    if quant_cfg.recipe_sweep_candidates:
        return custom_candidates(quant_cfg.recipe_sweep_candidates)
    return builtin_candidates(analysis, base_scheme=quant_cfg.scheme,
                              tuning_iters=quant_cfg.iters or 200)


def apply_candidate(base: QuantConfig, cand: Candidate) -> QuantConfig:
    """Produce the effective QuantConfig for one candidate."""
    return base.model_copy(update=cand.overrides)
