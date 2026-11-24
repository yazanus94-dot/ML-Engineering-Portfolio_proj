"""python -m infbench.run --batch 256 --repeats 7 --seed 0 -> reports/benchmark.{json,md}."""

import argparse
import sys
import traceback

from .bench import run, to_markdown, write_report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 512, 10])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--loops-batch",
        type=int,
        default=None,
        help="optional smaller batch for the pure-Python row; default: same as --batch",
    )
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)
    try:
        res = run(tuple(a.sizes), a.batch, a.repeats, a.seed, a.loops_batch)
        write_report(res, a.out)
    except Exception as exc:  # keep tracebacks out of committed reports
        print(f"benchmark failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    print(to_markdown(res))
    return 0 if all(r["within_tolerance"] for r in res["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
