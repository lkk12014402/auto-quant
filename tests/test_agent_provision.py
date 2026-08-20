from __future__ import annotations

import json

from aqa.agent import provision


def test_provision_lays_down_openclaw_and_copilot(tmp_path):
    provision.provision(home=tmp_path, minimax_key="KEY123", skills_dir=None)
    assert (tmp_path / ".openclaw" / "openclaw.json").is_file()
    auth = json.loads((tmp_path / ".openclaw/agents/main/agent/auth-profiles.json").read_text())
    for name in ("minimax:cn", "minimax:global", "minimax-global"):
        assert auth["profiles"][name]["key"] == "KEY123"
    assert (tmp_path / ".copilot" / "settings.json").is_file()


def test_provision_without_minimax_key_leaves_placeholder(tmp_path):
    provision.provision(home=tmp_path, minimax_key=None, skills_dir=None)
    auth = json.loads((tmp_path / ".openclaw/agents/main/agent/auth-profiles.json").read_text())
    assert auth["profiles"]["minimax:cn"]["key"].startswith("<")


def test_provision_copies_skills_to_both_backends(tmp_path):
    skills = tmp_path / "skills" / "quantize"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("recipe")
    provision.provision(home=tmp_path, minimax_key=None, skills_dir=tmp_path / "skills")
    assert (tmp_path / ".copilot/skills/quantize/SKILL.md").read_text() == "recipe"
    assert (tmp_path / ".openclaw/workspace/skills/quantize/SKILL.md").read_text() == "recipe"


def test_provision_does_not_clobber_existing_copilot_settings(tmp_path):
    dst = tmp_path / ".copilot"
    dst.mkdir()
    (dst / "settings.json").write_text('{"model": "custom"}')
    provision.provision(home=tmp_path, minimax_key=None, skills_dir=None)
    assert json.loads((dst / "settings.json").read_text())["model"] == "custom"


def test_provision_copilot_model_override(tmp_path):
    provision.provision(home=tmp_path, minimax_key=None, copilot_model="claude-sonnet-4.5",
                        skills_dir=None)
    assert json.loads((tmp_path / ".copilot/settings.json").read_text())["model"] == \
        "claude-sonnet-4.5"


def test_sync_minimax_key_missing_file(tmp_path):
    assert provision.sync_minimax_key(tmp_path / "nope.json", "K") is False


def test_main_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_MM_KEY", "envkey")
    rc = provision.main(["--home", str(tmp_path), "--minimax-env", "MY_MM_KEY"])
    assert rc == 0
    auth = json.loads((tmp_path / ".openclaw/agents/main/agent/auth-profiles.json").read_text())
    assert auth["profiles"]["minimax:cn"]["key"] == "envkey"
