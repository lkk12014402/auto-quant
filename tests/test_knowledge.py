from pathlib import Path

def test_references_push_and_manifest(tmp_path, monkeypatch):
    from aqa.agent.knowledge import KnowledgeStore

    src = tmp_path / "src"
    src.mkdir()
    (src / "auto_round_recipe.md").write_text("fresh knowledge")
    store = KnowledgeStore(local_dir=tmp_path / "kn", repo=None)
    monkeypatch.setattr(KnowledgeStore, "push_up", lambda self: True)
    ok = store.push_references(src, auto_round_ref="v0.15.0", note="seed")
    assert ok
    dest = store.references_dir
    assert (dest / "auto_round_recipe.md").read_text() == "fresh knowledge"
    import json
    m = json.loads((dest / "MANIFEST.json").read_text())
    assert m["auto_round_ref"] == "v0.15.0" and "auto_round_recipe.md" in m["files"]


def test_load_recipe_knowledge_prefers_synced(tmp_path):
    from aqa.agent.optimize import load_recipe_knowledge

    refs = tmp_path / "references"
    refs.mkdir()
    (refs / "auto_round_recipe.md").write_text("SYNCED_VERSION")
    text = load_recipe_knowledge(knowledge_dir=tmp_path)
    assert "SYNCED_VERSION" in text
    # packaged fallback when the synced copy is absent
    text2 = load_recipe_knowledge(knowledge_dir=tmp_path / "nonexistent")
    assert "SYNCED_VERSION" not in text2 and text2 != ""


def test_load_recipe_knowledge_use_synced_false_pins_packaged(tmp_path):
    from aqa.agent.optimize import load_recipe_knowledge

    refs = tmp_path / "references"
    refs.mkdir()
    (refs / "auto_round_recipe.md").write_text("SYNCED_VERSION")
    text = load_recipe_knowledge(knowledge_dir=tmp_path, use_synced=False)
    assert "SYNCED_VERSION" not in text and text != ""


def test_use_references_config_field():
    from aqa.config import KnowledgeConfig

    assert KnowledgeConfig().use_references is True
    assert KnowledgeConfig(use_references=False).use_references is False


def test_push_references_incremental_file(tmp_path, monkeypatch):
    import json as _json
    from aqa.agent.knowledge import KnowledgeStore

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("A v1")
    (src / "b.md").write_text("B v1")
    store = KnowledgeStore(local_dir=tmp_path / "kn", repo=None)
    monkeypatch.setattr(KnowledgeStore, "push_up", lambda self: True)
    store.push_references(src, auto_round_ref="v1", note="seed")

    (src / "b.md").write_text("B v2")  # 只改 b.md
    store.push_references(src, file="b.md", auto_round_ref="v2", note="b 更新")
    dest = store.references_dir
    assert (dest / "a.md").read_text() == "A v1"       # 未动
    assert (dest / "b.md").read_text() == "B v2"
    m = _json.loads((dest / "MANIFEST.json").read_text())
    assert m["files"]["a.md"]["auto_round_ref"] == "v1"   # 逐文件新鲜度独立
    assert m["files"]["b.md"]["auto_round_ref"] == "v2"
    assert m["auto_round_ref"] == "v2"


def test_knowledge_layout_yaml_loads():
    import yaml
    from pathlib import Path
    p = Path(__file__).parent.parent / "aqa" / "agent" / "knowledge_layout.yaml"
    data = yaml.safe_load(p.read_text())
    assert data["layout"] and "manifest_schema" in data
    refs = [e for e in data["layout"] if e["path"] == "references/"]
    assert refs and refs[0]["producer"] == "owner"


def test_merge_tree_never_overwrites_patch_bytes(tmp_path):
    from aqa.agent.knowledge import KnowledgeStore

    store = KnowledgeStore(local_dir=tmp_path / "local", repo=None)
    remote = tmp_path / "remote"
    local_patch = Path(store.local_dir) / "patches" / "quantize" / "abc" / "auto.patch"
    remote_patch = remote / "patches" / "quantize" / "abc" / "auto.patch"
    local_patch.parent.mkdir(parents=True)
    remote_patch.parent.mkdir(parents=True)
    local_patch.write_text("LOCAL FIX\n")
    remote_patch.write_text("REMOTE FIX\n")

    store._merge_tree(remote, Path(store.local_dir))

    assert local_patch.read_text() == "LOCAL FIX\n"
    assert (local_patch.with_name("auto.remote.patch")).read_text() == "REMOTE FIX\n"
