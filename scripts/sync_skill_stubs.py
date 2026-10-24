"""Generate harness-discoverable skill stubs from the canonical `skills/`.

External agents (Claude Code / Codex / Copilot CLI) auto-discover skills under
`.claude/skills/<name>/` and `.agents/skills/<name>/` by the `description`
frontmatter (the trigger surface). To avoid duplicating skill bodies, we ship
*thin stubs*: the verbatim frontmatter + a one-line pointer to the canonical
`skills/<name>/SKILL.md` (Quark's stub->impl pattern). Knowledge (recipes/,
docs/, cookbooks/) is NOT mirrored — it is reached on demand via `aqa recipe`
and path references inside the body.

Run from the repo root:  python scripts/sync_skill_stubs.py
Idempotent; safe to re-run after editing any skill's frontmatter.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
TARGETS = (".claude/skills", ".agents/skills")


def _frontmatter(text: str) -> str:
    """Return the verbatim YAML frontmatter block (including the `---` fences)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("skill missing frontmatter")
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    return "\n".join(lines[: end + 1])


def _stub(name: str, frontmatter: str) -> str:
    up = "../../.."  # .claude/skills/<name>/ -> repo root
    return (
        f"{frontmatter}\n\n"
        f"<!-- GENERATED STUB — do not edit. Source: skills/{name}/SKILL.md\n"
        f"     Regenerate: python scripts/sync_skill_stubs.py -->\n\n"
        f"**Skill body:** read [`skills/{name}/SKILL.md`]({up}/skills/{name}/SKILL.md), "
        f"and follow the interaction loop in "
        f"[`skills/CONTRACT.md`]({up}/skills/CONTRACT.md). "
        f"Route other requests via [`skills/SKILLS.md`]({up}/skills/SKILLS.md).\n"
    )


def main() -> None:
    skills = sorted(p for p in SKILLS.iterdir() if (p / "SKILL.md").is_file())
    written = []
    for skill in skills:
        name = skill.name
        frontmatter = _frontmatter((skill / "SKILL.md").read_text(encoding="utf-8"))
        body = _stub(name, frontmatter)
        for target in TARGETS:
            dest = ROOT / target / name / "SKILL.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
            written.append(dest.relative_to(ROOT))

    # Root pointer so Claude Code loads AGENTS.md (Codex/Copilot read it natively).
    claude_md = ROOT / "CLAUDE.md"
    claude_md.write_text("@AGENTS.md\n", encoding="utf-8")
    written.append(claude_md.relative_to(ROOT))

    print(f"synced {len(skills)} skills -> {len(TARGETS)} harness dirs")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
