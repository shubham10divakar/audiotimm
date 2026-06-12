"""audiotimm CLI — ``audiotimm predict <file> [--model <name>] [--top <k>]``"""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_predict(args: argparse.Namespace) -> None:
    from audiotimm import Classifier

    clf = Classifier.load(args.model or None, device=args.device)
    files = args.files

    if len(files) == 1:
        result = clf.predict(files[0])
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            for label, score in result.top(args.top):
                print(f"  {score:.4f}  {label}")
    else:
        results = clf.predict(files)
        for path, result in zip(files, results):
            print(f"\n{path}")
            if args.json:
                print(json.dumps(result.as_dict(), indent=2))
            else:
                for label, score in result.top(args.top):
                    print(f"  {score:.4f}  {label}")


def _cmd_list(args: argparse.Namespace) -> None:
    from audiotimm import registry

    specs = registry.list(wave=args.wave or None, task=args.task or None)
    if not specs:
        print("No models match the given filters.")
        return
    print(f"{'Name':<30}  {'Wave':<4}  {'SR':>6}  {'Classes':>7}  Description")
    print("-" * 80)
    for s in sorted(specs, key=lambda x: (x.wave, x.name)):
        print(f"{s.name:<30}  {s.wave:<4}  {s.sample_rate:>6}  {s.n_classes:>7}  {s.description}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audiotimm",
        description="audiotimm — The Model Hub for Audio Intelligence",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------- predict ----------
    p = sub.add_parser("predict", help="Classify one or more audio files")
    p.add_argument("files", nargs="+", help="Audio file path(s)")
    p.add_argument("-m", "--model", default=None, help="Model zoo-id (default: panns-cnn14)")
    p.add_argument("-k", "--top", type=int, default=5, help="Top-k results (default: 5)")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=_cmd_predict)

    # ---------- list ----------
    ls = sub.add_parser("list", help="List available models")
    ls.add_argument("--wave", default=None, help="Filter by wave (M0, M1, …)")
    ls.add_argument("--task", default=None, help="Filter by task (tagging, zero-shot, …)")
    ls.set_defaults(func=_cmd_list)

    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
