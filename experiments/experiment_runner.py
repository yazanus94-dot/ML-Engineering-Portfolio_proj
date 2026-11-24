"""Run and record a reproducible experiment from a JSON configuration.

The only built-in experiment is the synthetic NumPy spiral task. It is
explicitly *not* a biomedical experiment. The applied biomedical project has
its own runner and dataset under ``applied-biomedical-ml/``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "ml-numpy-from-scratch" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_spiral import train  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: dict) -> dict:
    """Validate the experiment schema and return the same dictionary."""
    _require(isinstance(config, dict), "configuration must be a JSON object")
    for key in ("experiment", "project", "dataset", "model", "seed", "hyperparameters"):
        _require(key in config, f"missing required field: {key}")
    _require(
        isinstance(config["experiment"], str) and config["experiment"].strip(),
        "experiment must be a non-empty string",
    )
    _require(
        isinstance(config["project"], str) and config["project"].strip(),
        "project must be a non-empty string",
    )
    _require(
        isinstance(config["seed"], int) and not isinstance(config["seed"], bool),
        "seed must be an integer",
    )
    _require(config["seed"] >= 0, "seed must be non-negative")

    dataset = config["dataset"]
    _require(isinstance(dataset, dict), "dataset must be an object")
    for key in ("name", "kind", "source", "license", "is_biomedical"):
        _require(key in dataset, f"dataset is missing required field: {key}")
    _require(
        dataset["kind"] in {"synthetic", "public", "private"}, "dataset.kind is invalid"
    )
    _require(
        isinstance(dataset["is_biomedical"], bool),
        "dataset.is_biomedical must be boolean",
    )
    _require(
        not (dataset["is_biomedical"] and dataset["kind"] == "synthetic"),
        "a synthetic dataset cannot be marked biomedical",
    )

    model = config["model"]
    _require(isinstance(model, dict), "model must be an object")
    for key in ("name", "implementation"):
        _require(key in model, f"model is missing required field: {key}")

    hp = config["hyperparameters"]
    _require(isinstance(hp, dict), "hyperparameters must be an object")
    for key in ("epochs", "hidden", "lr"):
        _require(key in hp, f"hyperparameters is missing required field: {key}")
    _require(
        isinstance(hp["epochs"], int) and hp["epochs"] > 0, "epochs must be positive"
    )
    _require(
        isinstance(hp["hidden"], int) and hp["hidden"] > 0, "hidden must be positive"
    )
    _require(
        isinstance(hp["lr"], (int, float))
        and not isinstance(hp["lr"], bool)
        and hp["lr"] > 0,
        "lr must be positive",
    )
    return config


def load_config(path) -> dict:
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        return validate_config(json.load(handle))


def _atomic_write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_experiment(config: dict) -> dict:
    """Execute the configured synthetic experiment and return a JSON-ready record."""
    validate_config(config)
    hp = config["hyperparameters"]
    started = time.perf_counter()
    raw = train(
        seed=config["seed"],
        epochs=hp["epochs"],
        hidden=hp["hidden"],
        lr=hp["lr"],
    )
    duration = time.perf_counter() - started
    metrics = {
        "first_loss": raw["first_loss"],
        "final_loss": raw["final_loss"],
        "train_accuracy": raw["train_accuracy"],
        "test_accuracy": raw["test_accuracy"],
    }
    return {
        "experiment": config["experiment"],
        "project": config["project"],
        "config": config,
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "command": config.get(
            "command",
            "python experiments/experiment_runner.py --config <config>",
        ),
        "duration_seconds": duration,
        "deterministic": True,
    }


def write_result(result: dict, path) -> pathlib.Path:
    path = pathlib.Path(path)
    _atomic_write(path, json.dumps(result, indent=2) + "\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = run_experiment(config)
        output = pathlib.Path(args.output or config["output"])
        if not output.is_absolute():
            output = ROOT / output
        write_result(result, output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"experiment failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
