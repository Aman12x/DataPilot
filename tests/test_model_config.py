from agents.analyze import node_shared


def test_model_defaults_to_fast_model(monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setenv("FAST_MODEL", "claude-haiku-4-5-20251001")

    assert node_shared._model() == "claude-haiku-4-5-20251001"


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("MODEL", "custom-model")

    assert node_shared._model() == "custom-model"
