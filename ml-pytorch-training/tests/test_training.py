"""Tests for the PyTorch training/serving lab. Optional deps skip, never fabricate."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pytorch_lab.benchmark_cpu import benchmark  # noqa: E402
from pytorch_lab.data import make_spirals, set_seed  # noqa: E402
from pytorch_lab.export_onnx import export_and_validate  # noqa: E402
from pytorch_lab.model import MLP, load_checkpoint  # noqa: E402
from pytorch_lab.train import seeded_split, train  # noqa: E402

TINY = {
    "input_dim": 2,
    "hidden": [8, 8],
    "num_classes": 2,
    "epochs": 2,
    "batch_size": 16,
    "lr": 0.01,
    "seed": 0,
    "n_per_class": 20,
}


def test_model_forward_shapes():
    model = MLP(2, (8,), 2)
    out = model(torch.randn(7, 2))
    assert out.shape == (7, 2)
    assert torch.isfinite(out).all()


def test_spirals_deterministic():
    set_seed(0)
    x1, y1 = make_spirals(n_per_class=12, seed=3)
    x2, y2 = make_spirals(n_per_class=12, seed=3)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert x1.shape == (24, 2) and y1.shape == (24,)


def test_seeded_split_contains_both_classes():
    x, y = make_spirals(n_per_class=20, seed=0)
    x_train, y_train, x_test, y_test = seeded_split(x, y, 0)
    assert set(y_train.tolist()) == {0, 1}
    assert set(y_test.tolist()) == {0, 1}


def test_training_generalizes_beyond_chance(tmp_path):
    config = dict(TINY, epochs=25, n_per_class=60, hidden=[16, 16])
    manifest = train(config, tmp_path)
    assert manifest["metrics"]["test_accuracy"] >= 0.75

    manifest = train(TINY, tmp_path)
    ckpt = tmp_path / "checkpoint.pt"
    exp = tmp_path / "experiment.json"
    assert ckpt.exists() and exp.exists()
    assert manifest["config"]["seed"] == 0
    assert 0 <= manifest["metrics"]["train_accuracy"] <= 1
    assert len(manifest["checksum_sha256"]) == 64
    model, checkpoint = load_checkpoint(ckpt)
    assert set(checkpoint) >= {"config", "state_dict", "metrics", "version"}
    assert model(torch.randn(3, 2)).shape == (3, 2)
    assert json.loads(exp.read_text())["checksum_sha256"] == manifest["checksum_sha256"]


def test_training_deterministic(tmp_path):
    first = train(TINY, tmp_path / "a")
    second = train(TINY, tmp_path / "b")
    assert first["metrics"] == second["metrics"]


def test_cpu_benchmark(tmp_path):
    train(TINY, tmp_path)
    result = benchmark(tmp_path / "checkpoint.pt", batch_size=8, repeats=3, warmup=1)
    assert result["batch_size"] == 8 and result["repeats"] == 3
    assert result["median_ms"] > 0 and result["samples_per_second"] > 0


def test_onnx_export_and_runtime_validation(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    train(TINY, tmp_path)
    report = export_and_validate(
        tmp_path / "checkpoint.pt", tmp_path / "model.onnx", tmp_path / "onnx.json"
    )
    assert report["status"] == "validated"
    assert report["passed"] is True
    assert report["argmax_agreement"] == 1.0


def test_fastapi_app_routes(tmp_path):
    pytest.importorskip("fastapi")
    train(TINY, tmp_path)
    app = __import__("pytorch_lab.serve", fromlist=["create_app"]).create_app(
        tmp_path / "checkpoint.pt"
    )
    routes = {route.path for route in app.routes}
    assert {"/health", "/predict"}.issubset(routes)
