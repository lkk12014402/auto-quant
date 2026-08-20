from aqa.registry import BACKENDS, EXECUTORS, HARNESSES, RECIPE_STORES, SCORERS, SINKS


def test_builtin_executors():
    assert set(EXECUTORS.names()) >= {"local", "ssh"}
    assert EXECUTORS.get("local").name == "local"


def test_builtin_harnesses():
    assert set(HARNESSES.names()) >= {"native", "lm_eval"}


def test_builtin_backends():
    assert set(BACKENDS.names()) >= {"hf", "vllm"}
    assert BACKENDS.create("hf").lm_eval_model_flag == "hf"


def test_builtin_sinks_and_stores():
    assert "local_fs" in SINKS.names()
    assert "repo_dir" in RECIPE_STORES.names()


def test_scorers_resolvable():
    assert set(SCORERS.names()) >= {"exact_match", "numeric_match"}
    fn = SCORERS.get("numeric_match")
    assert callable(fn)


def test_runtime_registration():
    HARNESSES.register("dummy", "aqa.eval.harness.native:NativeHarness")
    assert "dummy" in HARNESSES.names()


def test_recipe_store_roundtrip(tmp_path):
    from aqa.recipe_store import RepoDirStore

    store = RepoDirStore(root=str(tmp_path))
    store.pin("org/model", {"scheme": "MXFP4", "cmd": "auto-round ..."})
    loaded = store.load("org/model")
    assert loaded["scheme"] == "MXFP4"
