"""Recipe knowledge base: hot-start lookup + strict pin-back (RFC §3.4).

Quant recipes used to be *write-only* art: ``recipes/quant/*.json`` held rich,
hand-authored recipes and :class:`aqa.recipe_store.RecipeStore` exposed
``load``/``pin`` — but **nothing read or wrote them**.  This module wires that
dead code into the run loop:

* :meth:`RecipeKB.lookup` — **hot-start**: find the closest validated recipe for
  the model-under-quant by *progressive field relaxation*
  (model id → architecture / model_type → scheme), and surface its command +
  ignore rationale + past failures.  The caller injects it into the
  agent-optimize prompt as a *starting point* (failures are evidence, never a
  hard veto; the agent re-validates against the current box).  A hit also carries
  Hyperloom-style **trust + staleness** signals — ``authority``/``confidence`` and,
  when the live stack drifted from the one the recipe was validated on
  (``stack_fingerprint``), a ``stale`` flag + lowered confidence.
* :meth:`RecipeKB.pin` — **strict pin-back**: write a run's recipe back **only
  when it passed the gate**, into a separate ``auto/`` namespace so machine
  output never clobbers curated recipes.  A GO also stamps the validated
  ``stack_fingerprint``, bumps ``confidence`` per repeated GO, and appends a row
  to an append-only session ledger.
* :meth:`RecipeKB.record_outcome` — **negative knowledge** (Hyperloom
  ``what_failed``): a *non-GO* CLOSE appends a structured failure + session row to
  the same auto row **without** writing any validated ``schemes``, so the next run
  is warned ("NO-GO; scheme X dropped mean by Δ") while the KB never claims the
  failed config is trustworthy.  Outcomes are a closed vocabulary
  (:data:`ALLOWED_OUTCOMES`).

The lookup tolerates both the canonical shape :meth:`pin` writes and the richer
free-form curated recipes (it extracts what it can, best-effort).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_util import get_logger
from ..recipe_store import RepoDirStore, _QUANT_ROOT, _safe_key

_log = get_logger("agent.recipe_kb")

# Match quality (higher wins). Model-id hit beats architecture/family hit; a hit
# that carries the *requested* scheme beats one we have to relax the scheme for.
LEVEL_MODEL_SCHEME = 4
LEVEL_MODEL_ANY = 3
LEVEL_FAMILY_SCHEME = 2
LEVEL_FAMILY_ANY = 1

_MAX_LESSONS = 8
_MAX_FAILURES = 6
_MAX_SESSIONS = 20

# Closed outcome vocabulary (Hyperloom ``ALLOWED_OUTCOMES`` pattern): every CLOSE
# records exactly one of these.  Only ``GO`` pins validated schemes; the rest are
# recorded as *negative knowledge* (what failed) without ever granting trust.
OUTCOME_GO = "GO"
OUTCOME_NO_GO = "NO-GO"
OUTCOME_FAILED = "FAILED"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
ALLOWED_OUTCOMES = frozenset(
    {OUTCOME_GO, OUTCOME_NO_GO, OUTCOME_FAILED, OUTCOME_INCONCLUSIVE})

# Trust defaults (Hyperloom ``authority``/``confidence``).  A GO pin starts at
# ``_CONF_BASE`` and earns ``_CONF_STEP`` per repeated GO (capped); a stale stack
# multiplies the read-time confidence by ``_CONF_STALE_PENALTY``.
_CONF_BASE = 0.85
_CONF_STEP = 0.05
_CONF_CAP = 0.98
_CONF_STALE_PENALTY = 0.4

# Stack tools whose version drift makes a recipe stale (quant + eval).  Names are
# canonicalised (auto-round/auto_round -> auto_round, lm-eval/lm_eval -> lm_eval)
# so two runs on the same stack produce byte-identical fingerprints.
_FINGERPRINT_TOOLS = ("auto_round", "vllm", "lm_eval", "transformers", "torch")
_FINGERPRINT_ALIASES = {"auto-round": "auto_round", "lm-eval": "lm_eval"}


def _canon_tool(name: str) -> str:
    return _FINGERPRINT_ALIASES.get(name, name)


def _stack_fingerprint(env_lock: dict[str, Any] | None) -> dict[str, str]:
    """Compact ``{tool: version}`` (+ accelerator family) from an env_lock payload.

    Keeps only the quant+eval tools whose version drift can invalidate a recipe,
    so two runs on the same stack produce identical fingerprints — the basis for
    staleness detection.  Tolerant of a missing/partial ``packages`` map.
    """
    if not isinstance(env_lock, dict):
        return {}
    fp: dict[str, str] = {}
    packages = env_lock.get("packages")
    if isinstance(packages, dict):
        for raw, ver in packages.items():
            tool = _canon_tool(str(raw))
            if tool in _FINGERPRINT_TOOLS and ver:
                fp.setdefault(tool, str(ver))
    acc = env_lock.get("accelerator")
    if isinstance(acc, dict) and acc.get("family"):
        fp["accelerator"] = str(acc["family"])
    return fp


def _stale_tools(stored: Any, current: Any) -> list[str]:
    """Tools whose pinned version differs from the current stack.

    Compares only keys present in BOTH fingerprints — a missing probe is unknown,
    not stale — so best-effort gaps never raise a false staleness alarm.
    """
    if not isinstance(stored, dict) or not isinstance(current, dict):
        return []
    return [k for k in sorted(stored)
            if k in current and str(stored[k]) != str(current[k])]


def _sessions_summary(sessions: Any) -> str | None:
    """One-line outcome tally for a recipe's append-only session ledger."""
    if not isinstance(sessions, list) or not sessions:
        return None
    counts: dict[str, int] = {}
    for s in sessions:
        if isinstance(s, dict):
            outcome = str(s.get("outcome") or "?")
            counts[outcome] = counts.get(outcome, 0) + 1
    if not counts:
        return None
    return ", ".join(f"{n} {o}" for o, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def _append_session(entry: dict[str, Any], *, outcome: str, scheme: str | None,
                    mean_before: float | None = None, mean_after: float | None = None,
                    source_run: str | None = None,
                    stack_fingerprint: dict[str, str] | None = None,
                    note: str | None = None) -> None:
    """Append one append-only session row (Hyperloom ``SessionSummary``) to the
    auto ``entry``, bounded to the most recent ``_MAX_SESSIONS``."""
    from ..session._io import now_iso
    row: dict[str, Any] = {"date": now_iso(), "outcome": outcome}
    if scheme:
        row["scheme"] = scheme
    if source_run:
        row["run"] = source_run
    if mean_before is not None:
        row["mean_before"] = round(float(mean_before), 4)
    if mean_after is not None:
        row["mean_after"] = round(float(mean_after), 4)
    if mean_before is not None and mean_after is not None:
        row["drop"] = round(float(mean_before) - float(mean_after), 4)
    if stack_fingerprint:
        row["stack"] = dict(stack_fingerprint)
    if note:
        row["note"] = note
    sessions = entry.get("sessions")
    if not isinstance(sessions, list):
        sessions = []
    sessions.append(row)
    entry["sessions"] = sessions[-_MAX_SESSIONS:]


@dataclass
class RecipeHit:
    """One matched recipe, with the fields a hot-start prompt needs."""

    key: str                       # model id of the matched recipe
    match_level: int
    match_reason: str
    source: str                    # "curated" | "auto"
    source_path: str
    scheme: str | None             # the scheme whose command we surfaced
    command: str | None
    ignore_layers: str | None
    lessons: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    eval_parity: dict[str, Any] | None = None   # structured eval口径 (thinking, model_args)
    stack_fingerprint: dict[str, str] | None = None  # stack this recipe was validated on
    confidence: float | None = None             # read-time trust (lowered when stale)
    authority: str | None = None                # "curated" | "experiential"
    stale: bool = False                         # current stack drifted from the pinned one
    stale_tools: list[str] = field(default_factory=list)
    sessions: str | None = None                 # ledger tally, e.g. "2 GO, 1 NO-GO"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "match_level": self.match_level,
            "match_reason": self.match_reason,
            "source": self.source,
            "source_path": self.source_path,
            "scheme": self.scheme,
            "command": self.command,
            "ignore_layers": self.ignore_layers,
            "lessons": list(self.lessons),
            "failures": list(self.failures),
            "result": self.result,
            "eval_parity": self.eval_parity,
            "stack_fingerprint": self.stack_fingerprint,
            "confidence": self.confidence,
            "authority": self.authority,
            "stale": self.stale,
            "stale_tools": list(self.stale_tools),
            "sessions": self.sessions,
        }


def _model_slug(model: str) -> str:
    """Last path/repo component, lowercased — matches a local path to a HF id.

    ``/home/u/models/Qwen3.8-27B`` and ``Qwen/Qwen3.8-27B`` both slug to
    ``qwen3.8-27b`` so a recipe keyed by HF id matches a local checkpoint.
    """
    return model.rstrip("/").split("/")[-1].strip().lower()


def _family(recipe: dict[str, Any]) -> str | None:
    """Best-effort architecture family token (``model_type``)."""
    mt = recipe.get("model_type")
    if isinstance(mt, str) and mt.strip():
        return mt.strip().lower()
    arch = recipe.get("architecture")
    if isinstance(arch, str) and arch.strip():
        # Curated recipes store prose like "qwen3_5 (Qwen3_5ForConditionalGeneration…)".
        return arch.strip().split()[0].split("(")[0].strip().lower() or None
    archs = recipe.get("architectures")
    if isinstance(archs, list) and archs:
        return str(archs[0]).strip().lower() or None
    return None


def _find_command(subtree: Any) -> str | None:
    """First ``command`` string anywhere under a scheme subtree (recursive)."""
    if isinstance(subtree, dict):
        cmd = subtree.get("command")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip()
        for value in subtree.values():
            found = _find_command(value)
            if found:
                return found
    elif isinstance(subtree, list):
        for value in subtree:
            found = _find_command(value)
            if found:
                return found
    return None


def _collect_lessons(recipe: dict[str, Any], scheme: str | None) -> list[str]:
    out: list[str] = []
    for key in ("lessons",):
        val = recipe.get(key)
        if isinstance(val, list):
            out.extend(str(x) for x in val if x)
    rationale = recipe.get("ignore_layers_rationale")
    if isinstance(rationale, str) and rationale.strip():
        out.append(f"ignore rationale: {rationale.strip()}")
    prose_parity = _find_scalar(recipe, "eval_parity")  # curated prose口径 (advisory)
    if prose_parity:
        out.append(f"eval parity: {prose_parity}")
    blocker = recipe.get("blocker_fixed")
    if isinstance(blocker, dict) and blocker.get("fix"):
        out.append(f"known fix: {blocker['fix']}")
    schemes = recipe.get("schemes")
    if isinstance(schemes, dict) and scheme in schemes and isinstance(schemes[scheme], dict):
        node = schemes[scheme]
        for k in ("note", "gate", "when", "caveats", "expected"):
            v = _find_scalar(node, k)
            if v:
                out.append(f"{scheme} {k}: {v}")
    # De-dup preserving order, bound length.
    seen: set[str] = set()
    uniq = [x for x in out if not (x in seen or seen.add(x))]
    return uniq[:_MAX_LESSONS]


def _collect_failures(recipe: dict[str, Any], scheme: str | None) -> list[str]:
    out: list[str] = []
    val = recipe.get("failures")
    if isinstance(val, list):
        out.extend(str(x) for x in val if x)
    blocker = recipe.get("blocker_fixed")
    if isinstance(blocker, dict):
        if blocker.get("symptom"):
            out.append(f"past blocker: {blocker['symptom']}")
        if blocker.get("bad_artifact"):
            out.append(f"bad artifact: {blocker['bad_artifact']}")
    schemes = recipe.get("schemes")
    if isinstance(schemes, dict) and scheme in schemes:
        risk = _find_scalar(schemes[scheme], "risk")
        if risk:
            out.append(f"{scheme} risk: {risk}")
    seen: set[str] = set()
    uniq = [x for x in out if not (x in seen or seen.add(x))]
    return uniq[:_MAX_FAILURES]


def _find_scalar(subtree: Any, key: str) -> str | None:
    """First string value for ``key`` anywhere under a subtree (recursive)."""
    if isinstance(subtree, dict):
        v = subtree.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        for value in subtree.values():
            found = _find_scalar(value, key)
            if found:
                return found
    elif isinstance(subtree, list):
        for value in subtree:
            found = _find_scalar(value, key)
            if found:
                return found
    return None


def _structured_eval_parity(recipe: dict[str, Any]) -> dict[str, Any] | None:
    """A machine-applicable eval口径 (thinking + model_args), if the recipe has one.

    Only a top-level ``eval_parity`` *object* is auto-applied to the eval phase —
    that is what :meth:`RecipeKB.pin` writes.  Curated prose
    (``verification.eval_parity`` as a string) is surfaced as an advisory lesson
    by :func:`_collect_lessons` instead, never auto-applied.
    """
    ep = recipe.get("eval_parity")
    return ep if isinstance(ep, dict) and ep else None


@dataclass
class _Entry:
    model: str
    slug: str
    family: str | None
    schemes: list[str]
    recipe: dict[str, Any]
    source: str
    path: Path


class RecipeKB:
    """Read/write access to validated quant recipes, keyed by model identity."""

    def __init__(self, curated_root: Path, auto_root: Path):
        self.curated_root = Path(curated_root)
        self.auto_root = Path(auto_root)
        self._auto_store = RepoDirStore(root=str(self.auto_root))

    @classmethod
    def from_config(cls, cfg: Any) -> "RecipeKB":
        root = Path(cfg.root).expanduser() if getattr(cfg, "root", None) else _QUANT_ROOT
        auto = root / getattr(cfg, "auto_subdir", "auto")
        return cls(curated_root=root, auto_root=auto)

    # ── read side (hot-start) ────────────────────────────────────────────────
    def _iter_entries(self) -> list[_Entry]:
        entries: list[_Entry] = []
        sources = [(self.curated_root, "curated"), (self.auto_root, "auto")]
        for root, source in sources:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.json")):
                # The auto namespace is a *subdir* of curated_root; don't double-read.
                if source == "curated" and path.parent != self.curated_root:
                    continue
                try:
                    recipe = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    _log.warning("recipe_kb: skipping unreadable %s (%s)", path, exc)
                    continue
                if not isinstance(recipe, dict):
                    continue
                model = str(recipe.get("model") or path.stem.replace("__", "/"))
                schemes = recipe.get("schemes")
                scheme_names = list(schemes.keys()) if isinstance(schemes, dict) else []
                entries.append(_Entry(
                    model=model, slug=_model_slug(model), family=_family(recipe),
                    schemes=scheme_names, recipe=recipe, source=source, path=path))
        return entries

    def entries_summary(self) -> list[dict[str, Any]]:
        """One row per known recipe (for ``aqa recipe list``)."""
        return [
            {"model": e.model, "family": e.family, "schemes": e.schemes,
             "source": e.source, "path": str(e.path)}
            for e in self._iter_entries()
        ]

    def show(self, model: str) -> dict[str, Any] | None:
        """Raw recipe dict for an exact model-id match (for ``aqa recipe show``)."""
        slug = _model_slug(model)
        best: _Entry | None = None
        for entry in self._iter_entries():
            if entry.slug != slug:
                continue
            # Curated outranks auto at the same model id.
            if best is None or (entry.source == "curated" and best.source != "curated"):
                best = entry
        return best.recipe if best is not None else None

    def lookup(self, model: str, *, scheme: str | None = None,
               model_type: str | None = None,
               current_stack: dict[str, str] | None = None) -> RecipeHit | None:
        """Closest validated recipe via progressive relaxation, or ``None``.

        Order: exact model id + requested scheme → exact model id (any scheme) →
        same architecture family + scheme → same family (any scheme).  Curated
        recipes outrank auto-pinned ones at the same level.  When ``current_stack``
        is given, the hit is annotated with staleness (pinned stack drifted) and a
        read-time ``confidence`` lowered accordingly.
        """
        slug = _model_slug(model)
        fam = model_type.strip().lower() if model_type else None
        best: tuple[int, int, _Entry, str | None] | None = None  # (level, source_rank, entry, scheme)

        for entry in self._iter_entries():
            has_scheme = scheme is not None and scheme in entry.schemes
            level = 0
            if entry.slug == slug:
                level = LEVEL_MODEL_SCHEME if has_scheme else LEVEL_MODEL_ANY
            elif fam is not None and entry.family == fam:
                level = LEVEL_FAMILY_SCHEME if has_scheme else LEVEL_FAMILY_ANY
            if level == 0:
                continue
            source_rank = 1 if entry.source == "curated" else 0
            pick_scheme = scheme if has_scheme else (entry.schemes[0] if entry.schemes else None)
            cand = (level, source_rank, entry, pick_scheme)
            if best is None or (level, source_rank) > (best[0], best[1]):
                best = cand

        if best is None:
            return None
        level, _rank, entry, pick_scheme = best
        reason = {
            LEVEL_MODEL_SCHEME: f"exact model + scheme {pick_scheme}",
            LEVEL_MODEL_ANY: f"exact model (relaxed scheme → {pick_scheme})",
            LEVEL_FAMILY_SCHEME: f"same family {entry.family} + scheme {pick_scheme}",
            LEVEL_FAMILY_ANY: f"same family {entry.family} (relaxed scheme → {pick_scheme})",
        }[level]
        schemes = entry.recipe.get("schemes")
        subtree = schemes.get(pick_scheme) if isinstance(schemes, dict) and pick_scheme else None
        command = _find_command(subtree) if subtree is not None else None
        ignore = entry.recipe.get("ignore_layers")
        if not isinstance(ignore, str):
            ignore = _find_scalar(subtree, "ignore_layers") if subtree is not None else None
        result = entry.recipe.get("result")
        if not isinstance(result, dict) and isinstance(subtree, dict):
            result = subtree.get("result") if isinstance(subtree.get("result"), dict) else None
        stored_fp = entry.recipe.get("stack_fingerprint")
        stored_fp = stored_fp if isinstance(stored_fp, dict) else None
        drift = _stale_tools(stored_fp, current_stack)
        stale = bool(drift)
        authority = ("curated" if entry.source == "curated"
                     else str(entry.recipe.get("authority") or "experiential"))
        has_validated = isinstance(entry.recipe.get("schemes"), dict) and bool(entry.recipe["schemes"])
        confidence = entry.recipe.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = _CONF_CAP if entry.source == "curated" else (
                _CONF_BASE if has_validated else 0.0)
        confidence = float(confidence)
        if stale:
            confidence = round(confidence * _CONF_STALE_PENALTY, 3)
        return RecipeHit(
            key=entry.model, match_level=level, match_reason=reason,
            source=entry.source, source_path=str(entry.path), scheme=pick_scheme,
            command=command, ignore_layers=ignore if isinstance(ignore, str) else None,
            lessons=_collect_lessons(entry.recipe, pick_scheme),
            failures=_collect_failures(entry.recipe, pick_scheme),
            result=result if isinstance(result, dict) else None,
            eval_parity=_structured_eval_parity(entry.recipe),
            stack_fingerprint=stored_fp, confidence=confidence, authority=authority,
            stale=stale, stale_tools=drift,
            sessions=_sessions_summary(entry.recipe.get("sessions")))

    @staticmethod
    def as_hot_start_context(hit: RecipeHit) -> str:
        """Compact prompt block seeding agent-optimize with a validated recipe."""
        lines = [
            "## Hot-start recipe (Recipe KB — ADVISORY, not a hard rule)",
            f"A validated recipe for a related target was found: {hit.match_reason} "
            f"(source: {hit.source}).",
            "Use it as your STARTING POINT: adapt `--model_name` / `--output_dir` to "
            "THIS run's paths, keep the proven precision/ignore decisions unless the "
            "current model structure contradicts them, and RE-VALIDATE — do not copy blindly.",
        ]
        trust: list[str] = []
        if hit.authority:
            trust.append(f"authority={hit.authority}")
        if hit.confidence is not None:
            trust.append(f"confidence={hit.confidence:.2f}")
        if hit.sessions:
            trust.append(f"history: {hit.sessions}")
        if trust:
            lines.append("- trust: " + ", ".join(trust))
        if hit.stale:
            lines.append(
                "- ⚠ STALE: the stack drifted since this recipe was validated "
                f"({', '.join(hit.stale_tools)}). Treat its numbers as approximate "
                "and RE-VALIDATE on the current stack before trusting the gate.")
        if hit.scheme:
            lines.append(f"- scheme: {hit.scheme}")
        if hit.command:
            lines.append(f"- validated command (adapt paths): {hit.command}")
        if hit.ignore_layers:
            lines.append(f"- ignore_layers: {hit.ignore_layers}")
        if hit.eval_parity:
            lines.append(f"- eval parity (validated口径): {hit.eval_parity}")
        if hit.lessons:
            lines.append("- lessons:")
            lines.extend(f"    - {x}" for x in hit.lessons)
        if hit.failures:
            lines.append("- past failures (evidence — avoid repeating, NOT a veto):")
            lines.extend(f"    - {x}" for x in hit.failures)
        return "\n".join(lines)

    # ── write side (strict pin-back) ─────────────────────────────────────────
    def pin(self, *, model: str, scheme: str, command: str,
            model_type: str | None = None, architecture: str | None = None,
            result: dict[str, Any] | None = None, gate_verdict: str | None = None,
            ignore_layers: str | None = None, lessons: list[str] | None = None,
            failures: list[str] | None = None, eval_parity: dict[str, Any] | None = None,
            source_run: str | None = None, stack_fingerprint: dict[str, str] | None = None,
            mean_before: float | None = None, mean_after: float | None = None) -> str | None:
        """Persist a *gate-passing* recipe to the ``auto/`` namespace (merging).

        Returns the written path, or ``None`` when the run did not pass the gate
        (strict: the KB only accumulates recipes we trust).  Merges into any
        existing auto entry for the model: adds/overwrites this scheme and unions
        lessons/failures, so repeated runs enrich one row instead of fragmenting.
        ``eval_parity`` (structured thinking/model_args口径) is stored at the row
        top level so the next run's eval phase can replay it deterministically.
        A GO also stamps the validated ``stack_fingerprint`` (for later staleness
        detection), bumps ``confidence`` per repeated GO, and appends a session row
        to the append-only ledger (Hyperloom ``authority``/``confidence``/sessions).
        """
        if gate_verdict is not None and gate_verdict.upper() != "GO":
            _log.info("recipe_kb: not pinning %s (gate=%s, only GO is pinned)",
                      model, gate_verdict)
            return None
        key = _safe_key(_model_slug(model))
        existing = self._auto_store.load(key)
        entry: dict[str, Any] = existing if isinstance(existing, dict) else {}
        entry.setdefault("model", model)
        if model_type:
            entry["model_type"] = model_type
        if architecture:
            entry["architecture"] = architecture
        from ..session._io import now_iso
        entry["validated_on"] = now_iso()
        if source_run:
            entry["source_run"] = source_run
        schemes = entry.get("schemes")
        if not isinstance(schemes, dict):
            schemes = {}
        scheme_node: dict[str, Any] = {"command": command}
        if ignore_layers:
            scheme_node["ignore_layers"] = ignore_layers
        if result:
            scheme_node["result"] = result
        if gate_verdict:
            scheme_node["gate"] = gate_verdict
        schemes[scheme] = scheme_node
        entry["schemes"] = schemes
        if ignore_layers:
            entry["ignore_layers"] = ignore_layers
        entry["lessons"] = _merge_list(entry.get("lessons"), lessons)
        entry["failures"] = _merge_list(entry.get("failures"), failures)
        if eval_parity:
            entry["eval_parity"] = eval_parity
        # Trust + staleness provenance (Hyperloom): a GO is experiential evidence
        # whose confidence grows with each independent GO on this row.
        entry["authority"] = "experiential"
        if stack_fingerprint:
            entry["stack_fingerprint"] = dict(stack_fingerprint)
            entry["last_profiled"] = entry["validated_on"]
        prior = entry.get("sessions")
        prior_go = sum(1 for s in prior if isinstance(s, dict) and s.get("outcome") == OUTCOME_GO) \
            if isinstance(prior, list) else 0
        entry["confidence"] = round(min(_CONF_CAP, _CONF_BASE + _CONF_STEP * prior_go), 3)
        _append_session(entry, outcome=OUTCOME_GO, scheme=scheme,
                        mean_before=mean_before, mean_after=mean_after,
                        source_run=source_run, stack_fingerprint=stack_fingerprint)
        return self._auto_store.pin(key, entry)

    def record_outcome(self, *, model: str, scheme: str | None, outcome: str,
                       model_type: str | None = None, architecture: str | None = None,
                       result: dict[str, Any] | None = None,
                       mean_before: float | None = None, mean_after: float | None = None,
                       reasons: list[str] | None = None, source_run: str | None = None,
                       stack_fingerprint: dict[str, str] | None = None) -> str | None:
        """Record a *non-GO* CLOSE as negative knowledge (Hyperloom ``what_failed``).

        Appends a structured failure line + an append-only session row to the
        model's auto row WITHOUT writing any validated ``schemes`` — so the next
        run is warned ("NO-GO; scheme X dropped mean by Δ → ...") while the KB
        never claims the failed config is trustworthy.  ``GO`` is rejected here
        (use :meth:`pin`); an unknown outcome raises ``ValueError``.  Returns the
        written path.
        """
        if outcome not in ALLOWED_OUTCOMES:
            raise ValueError(
                f"unknown outcome {outcome!r}; allowed: {sorted(ALLOWED_OUTCOMES)}")
        if outcome == OUTCOME_GO:
            raise ValueError("record_outcome is for non-GO outcomes; use pin() for GO")
        key = _safe_key(_model_slug(model))
        existing = self._auto_store.load(key)
        entry: dict[str, Any] = existing if isinstance(existing, dict) else {}
        entry.setdefault("model", model)
        if model_type:
            entry["model_type"] = model_type
        if architecture:
            entry["architecture"] = architecture
        from ..session._io import now_iso
        entry["updated_on"] = now_iso()
        if source_run:
            entry["last_run"] = source_run
        head_bits = [outcome]
        if scheme:
            head_bits.append(f"scheme {scheme}")
        if mean_before is not None and mean_after is not None:
            head_bits.append(
                f"mean {mean_before:.4f}->{mean_after:.4f} (Δ{mean_after - mean_before:+.4f})")
        head = "; ".join(head_bits)
        detail = " | ".join(str(r) for r in (reasons or []) if r)
        line = f"{head}: {detail}" if detail else head
        entry["failures"] = _merge_list(entry.get("failures"), [line])
        _append_session(entry, outcome=outcome, scheme=scheme,
                        mean_before=mean_before, mean_after=mean_after,
                        source_run=source_run, stack_fingerprint=stack_fingerprint,
                        note=detail or None)
        # Negative knowledge never grants trust: leave schemes/authority/confidence
        # untouched so a prior GO on this row keeps its validated standing.
        return self._auto_store.pin(key, entry)


def _merge_list(existing: Any, new: list[str] | None) -> list[str]:
    out: list[str] = list(existing) if isinstance(existing, list) else []
    for x in new or []:
        if x and x not in out:
            out.append(x)
    return out
