"""Reproducible CPU training with checkpointing and an experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from . import __version__
from .data import make_spirals, set_seed
from .model import MLP


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean())


def seeded_split(
    x: torch.Tensor, y: torch.Tensor, seed: int, train_fraction: float = 0.8
):
    """Seeded permutation before splitting; both classes occur in each partition."""
    permutation = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    x, y = x[permutation], y[permutation]
    split = int(train_fraction * len(x))
    return x[:split], y[:split], x[split:], y[split:]


def train(config: dict, output_dir: str | Path) -> dict:
    """Train, checkpoint and write a deterministic experiment manifest."""
    set_seed(int(config["seed"]))
    x, y = make_spirals(
        n_per_class=int(config["n_per_class"]), seed=int(config["seed"])
    )
    x_train, y_train, x_test, y_test = seeded_split(x, y, int(config["seed"]))

    model = MLP(config["input_dim"], tuple(config["hidden"]), config["num_classes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    criterion = nn.CrossEntropyLoss()
    batch_size = int(config["batch_size"])

    for _ in range(int(config["epochs"])):
        model.train()
        for start in range(0, len(x_train), batch_size):
            batch_x, batch_y = (
                x_train[start : start + batch_size],
                y_train[start : start + batch_size],
            )
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = accuracy(model(x_train), y_train)
        test_acc = accuracy(model(x_test), y_test)
        final_loss = float(criterion(model(x_train), y_train))
    metrics = {
        "final_train_loss": round(final_loss, 8),
        "train_accuracy": round(train_acc, 6),
        "test_accuracy": round(test_acc, 6),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    torch.save(
        {
            "version": __version__,
            "config": config,
            "state_dict": model.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )

    manifest = {
        "project": "ml-pytorch-training",
        "project_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m pytorch_lab.train",
        "config": config,
        "metrics": metrics,
        "checksum_sha256": _sha256(checkpoint_path),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "data_statement": "synthetic seeded two-spirals; not real data",
    }
    (out / "experiment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-per-class", type=int, default=250)
    parser.add_argument("--output", default="reports")
    args = parser.parse_args(argv)
    config = {
        "input_dim": 2,
        "hidden": [32, 32],
        "num_classes": 2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "n_per_class": args.n_per_class,
    }
    manifest = train(config, args.output)
    print(json.dumps(manifest["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
