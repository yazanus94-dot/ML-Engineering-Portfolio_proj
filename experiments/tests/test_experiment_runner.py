import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.experiment_runner import (  # noqa: E402
    load_config,
    main,
    run_experiment,
    validate_config,
    write_result,
)

CONFIG = ROOT / "experiments" / "configs" / "spiral_numpy.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_config_loads_and_validates():
    config = load_config(CONFIG)
    assert config["dataset"]["is_biomedical"] is False
    assert config["dataset"]["kind"] == "synthetic"
    assert config["hyperparameters"]["epochs"] > 0


@pytest.mark.parametrize(
    "missing", ["experiment", "dataset", "model", "seed", "hyperparameters"]
)
def test_missing_required_field_is_rejected(missing):
    config = _config()
    del config[missing]
    with pytest.raises(ValueError, match="missing required field"):
        validate_config(config)


def test_synthetic_dataset_cannot_be_biomedical():
    config = _config()
    config["dataset"]["is_biomedical"] = True
    with pytest.raises(
        ValueError, match="synthetic dataset cannot be marked biomedical"
    ):
        validate_config(config)


def test_invalid_seed_and_hyperparameters_are_rejected():
    config = _config()
    config["seed"] = -1
    with pytest.raises(ValueError, match="seed"):
        validate_config(config)

    config = _config()
    config["hyperparameters"]["epochs"] = 0
    with pytest.raises(ValueError, match="epochs"):
        validate_config(config)

    config = _config()
    config["hyperparameters"]["lr"] = 0.0
    with pytest.raises(ValueError, match="lr"):
        validate_config(config)


def test_run_is_deterministic():
    config = _config()
    a = run_experiment(config)
    b = run_experiment(config)
    assert a["metrics"] == b["metrics"]
    assert a["experiment"] == "spiral_numpy_seed0"
    assert 0.0 <= a["metrics"]["test_accuracy"] <= 1.0
    assert a["metrics"]["final_loss"] < a["metrics"]["first_loss"]


def test_write_result_creates_valid_json(tmp_path):
    result = run_experiment(_config())
    path = write_result(result, tmp_path / "results" / "run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"] == result["metrics"]
    assert not path.with_name(path.name + ".tmp").exists()


def test_cli_writes_requested_output(tmp_path):
    target = tmp_path / "out.json"
    code = main(["--config", str(CONFIG), "--output", str(target)])
    assert code == 0
    assert (
        json.loads(target.read_text(encoding="utf-8"))["experiment"]
        == "spiral_numpy_seed0"
    )


def test_cli_reports_invalid_config_without_writing(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"experiment": "x"}', encoding="utf-8")
    target = tmp_path / "never.json"
    assert main(["--config", str(bad), "--output", str(target)]) == 1
    assert not target.exists()
