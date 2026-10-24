"""PolicyGate: deterministic named-rule review of an agent's captured source edits.

RFC §3.5 asks that risky source edits go through a gate — isolated snapshot →
unified diff → **policy check** → re-test → pass becomes the new baseline, and a
rejection must *name the rule* in ``events.jsonl``.  :class:`aqa.agent.patch.PatchCapture`
already provides the snapshot/diff/restore mechanics; this module is the missing
**policy check**: it inspects the captured unified diff and flags edits that would
let a fix "pass" dishonestly or endanger the environment.

The gate is deterministic and content-only (it never runs the diff), so it is
fully unit-testable.  Enforcement (reverting a violating edit) is the caller's
job via ``PatchCapture.restore`` — the gate only decides and names rules.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ParsedDiff:
    """The parts of a unified diff the rules reason over."""

    paths: list[str] = field(default_factory=list)   # b/<label>/<rel> targets touched
    added: list[str] = field(default_factory=list)   # added content lines (no leading +)
    removed: list[str] = field(default_factory=list)  # removed content lines (no leading -)

    @property
    def net_deletions(self) -> int:
        return len(self.removed) - len(self.added)


def parse_diff(diff_text: str) -> ParsedDiff:
    """Extract touched paths + added/removed content from a unified diff."""
    pd = ParsedDiff()
    for line in (diff_text or "").splitlines():
        if line.startswith("+++ "):
            pd.paths.append(line[4:].strip())
        elif line.startswith("--- "):
            continue
        elif line.startswith("+"):
            pd.added.append(line[1:])
        elif line.startswith("-"):
            pd.removed.append(line[1:])
    return pd


@dataclass
class Violation:
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


@dataclass
class PolicyDecision:
    allowed: bool
    violations: list[Violation] = field(default_factory=list)

    @property
    def rule_names(self) -> list[str]:
        return [v.rule for v in self.violations]

    def to_event(self) -> dict:
        return {
            "allowed": self.allowed,
            "violations": [{"rule": v.rule, "detail": v.detail} for v in self.violations],
        }


@dataclass
class PolicyRule:
    """A named deny-rule: ``check`` returns a detail string when violated, else ""."""

    name: str
    description: str
    check: Callable[[ParsedDiff], str]


# ── default deny-rules (the "拒绝点名规则") ──────────────────────────────────

_TEST_PATH_RE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")
_SKIP_RE = re.compile(
    r"@(pytest\.mark\.(skip|xfail)|unittest\.skip)\b|\bpytest\.skip\s*\(|\.skipTest\s*\(")
_CUDA_DISABLE_RE = re.compile(
    r"torch\.cuda\.is_available\s*=|"
    r"CUDA_VISIBLE_DEVICES['\"]?\s*[:=]\s*['\"]\s*['\"]|"
    r"device\s*=\s*['\"]cpu['\"]")
_VCS_CI_RE = re.compile(r"(^|/)(\.github/|\.gitlab-ci|\.git/|\.circleci/)")


def _rule_edit_tests(pd: ParsedDiff) -> str:
    hits = [p for p in pd.paths if p != "/dev/null" and _TEST_PATH_RE.search(p)]
    return f"edits test files ({', '.join(sorted(set(hits))[:3])})" if hits else ""


def _rule_skip_tests(pd: ParsedDiff) -> str:
    hits = [ln.strip() for ln in pd.added if _SKIP_RE.search(ln)]
    return f"adds test skip/xfail ({hits[0][:80]})" if hits else ""


def _rule_disable_cuda(pd: ParsedDiff) -> str:
    hits = [ln.strip() for ln in pd.added if _CUDA_DISABLE_RE.search(ln)]
    return f"disables/forces-off CUDA ({hits[0][:80]})" if hits else ""


def _rule_touches_vcs_ci(pd: ParsedDiff) -> str:
    hits = [p for p in pd.paths if p != "/dev/null" and _VCS_CI_RE.search(p)]
    return f"edits VCS/CI config ({', '.join(sorted(set(hits))[:3])})" if hits else ""


def default_rules(max_net_deletions: int = 200) -> list[PolicyRule]:
    def _rule_mass_deletion(pd: ParsedDiff) -> str:
        if pd.net_deletions > max_net_deletions:
            return (f"net deletion {pd.net_deletions} lines exceeds {max_net_deletions} "
                    "(guts code rather than fixing it)")
        return ""

    return [
        PolicyRule("edit_tests",
                   "editing test files can make the verifier pass dishonestly",
                   _rule_edit_tests),
        PolicyRule("skip_tests",
                   "skip/xfail markers suppress a failure instead of fixing it",
                   _rule_skip_tests),
        PolicyRule("disable_cuda",
                   "stubbing CUDA / forcing CPU hides a real GPU regression",
                   _rule_disable_cuda),
        PolicyRule("touches_vcs_ci",
                   "a source fix must not touch VCS/CI configuration",
                   _rule_touches_vcs_ci),
        PolicyRule("mass_deletion",
                   "deleting large swaths of code to pass is not a fix",
                   _rule_mass_deletion),
    ]


class PolicyGate:
    """Review a captured unified diff against deny-rules, naming any violation."""

    def __init__(self, rules: list[PolicyRule] | None = None, *,
                 enforce: bool = False, max_net_deletions: int = 200):
        self.rules = rules if rules is not None else default_rules(max_net_deletions)
        self.enforce = enforce

    def review(self, diff_text: str) -> PolicyDecision:
        if not (diff_text or "").strip():
            return PolicyDecision(allowed=True)
        pd = parse_diff(diff_text)
        violations = [Violation(rule.name, detail)
                      for rule in self.rules
                      if (detail := rule.check(pd))]
        return PolicyDecision(allowed=not violations, violations=violations)
