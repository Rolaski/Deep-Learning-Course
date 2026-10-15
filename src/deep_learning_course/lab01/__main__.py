"""Run Lab 01: python -m deep_learning_course.lab01 [--fast] [--show]."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", default=os.environ.get("DL_FAST") == "1")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/lab01"))
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    from deep_learning_course.environment import check_environment
    from .experiments import run_experiments

    environment = check_environment()
    print(json.dumps(environment, indent=2), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results = run_experiments(args.output, fast=args.fast, show=args.show)
    results["environment"] = environment
    results["fast"] = args.fast
    (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
