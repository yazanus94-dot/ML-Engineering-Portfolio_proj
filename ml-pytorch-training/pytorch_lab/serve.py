"""Optional FastAPI model service — never imported by the training path."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import load_checkpoint


def create_app(checkpoint_path: str | Path):
    """Build a FastAPI app with /health and /predict routes."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    model, checkpoint = load_checkpoint(checkpoint_path)

    class Payload(BaseModel):
        features: list[list[float]]

    app = FastAPI(title="SleepPSG-style MLP service demo", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": checkpoint["version"]}

    @app.post("/predict")
    def predict(payload: Payload) -> dict:
        tensor = torch.tensor(payload.features, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            return {"error": "expected features of shape [batch, 2]"}
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)
        return {"probabilities": probabilities.tolist()}

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="reports/checkpoint.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(args.checkpoint), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
