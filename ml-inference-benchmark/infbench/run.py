"""python -m infbench.run --batch 256 --repeats 7 --seed 0   ->  reports/benchmark.{json,md}"""
import argparse, json, pathlib
from .bench import run, to_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 512, 10])
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--loops-batch", type=int, default=8)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    res = run(tuple(a.sizes), a.batch, a.repeats, a.seed, a.loops_batch)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(res, indent=2)); (out / "benchmark.md").write_text(to_markdown(res))
    print(to_markdown(res))
    return 0 if all(r["within_tolerance"] for r in res["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
