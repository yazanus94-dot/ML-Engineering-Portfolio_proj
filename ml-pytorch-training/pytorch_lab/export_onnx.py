"""ONNX export with numerical inference validation (optional dependencies)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .data import make_spirals
from .model import load_checkpoint


def export_and_validate(checkpoint_path: str | Path, output_path: str | Path,
                        report_path: str | Path | None = None) -> dict:
    """Export to ONNX and compare ONNX Runtime output against PyTorch logits."""
    report: dict = {"status": "skipped", "reason": None}
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
        import onnxscript  # noqa: F401  # required by recent torch exporters
    except ImportError as exc:
        report["reason"] = f"optional dependency unavailable: {exc.name}"
        if report_path:
            Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    _, _ = make_spirals(n_per_class=4, seed=int(config.get("seed", 0)))
    x, _ = make_spirals(n_per_class=4, seed=int(config.get("seed", 0)))
    sample = x[:8]

    onnx_path = Path(output_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Some torch exporters print non-ASCII progress symbols; Windows consoles
    # using cp1252 otherwise raise UnicodeEncodeError during export.
    stdout = getattr(sys, "stdout", None)
    if stdout is not None and hasattr(stdout, "reconfigure"):
        try:
            stdout.reconfigure(errors="replace")
        except (OSError, ValueError):
            pass
    torch.onnx.export(
        model, sample, str(onnx_path),
        input_names=["input"], output_names=["logits"], opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input": sample.numpy()})[0]
    with torch.no_grad():
        torch_logits = model(sample).numpy()
    max_abs_diff = float(np.max(np.abs(onnx_logits - torch_logits)))
    report = {
        "status": "validated",
        "onnx_file": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "opset": 13,
        "max_abs_logit_difference": max_abs_diff,
        "tolerance": 1e-5,
        "passed": bool(max_abs_diff <= 1e-5),
        "argmax_agreement": float(np.mean(
            np.argmax(onnx_logits, axis=1) == np.argmax(torch_logits, axis=1)
        )),
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="reports/checkpoint.pt")
    parser.add_argument("--output", default="reports/model.onnx")
    parser.add_argument("--report", default="reports/onnx_validation.json")
    args = parser.parse_args(argv)
    result = export_and_validate(args.checkpoint, args.output, args.report)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("passed", result["status"] == "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
