#!/usr/bin/env python3
"""Fit one classical method on prepared training features; evaluate validation."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("knn", "linear_svm", "random_forest"), default="knn")
    parser.add_argument("--implementation", choices=("reference", "student"), default="reference")
    parser.add_argument("--data", type=Path, default=Path("data/classical-mini"))
    parser.add_argument("--config", type=Path, help="Defaults to configs/<method>.json")
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    args = parser.parse_args(argv)
    try:
        from team.runner import run_experiment
        result = run_experiment(method=args.method, implementation=args.implementation,
                                data_dir=args.data, config_path=args.config, output_dir=args.output)
    except ImportError as error:
        print(f"Setup error: {error}. Install the CPU dependencies with "
              "'python -m pip install -r requirements-classical.txt'.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Run failed: {error}", file=sys.stderr)
        return 1
    metrics = result["metrics"]
    print(f"Completed {result['method']} / {result['implementation']['effective']} "
          f"on {result['data']['profile']} validation ({metrics['samples']} clips).")
    print(f"Top-1 {100 * metrics['top1']:.2f}% | Top-5 {100 * metrics['top5']:.2f}% | "
          f"Macro Top-1 {100 * metrics['macro_top1']:.2f}% | Macro Top-5 {100 * metrics['macro_top5']:.2f}%")
    print(f"Results: {args.output / 'run.json'}")
    if result["implementation"]["effective"] == "student_reference_delegate":
        print("This student template delegates to the reference implementation; it is not an independent student algorithm.")
    if result["data"]["profile"] == "mini":
        print("This is a restricted mini vocabulary; do not compare it directly with full 2,731-class scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
