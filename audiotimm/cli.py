"""audiotimm CLI — full command-line interface.

Commands
--------
  predict     Classify one or more audio files
  embed       Extract embeddings from audio files
  list        List all registered models
  info        Show detailed info for one model
  benchmark   Time inference for a model on a file

Global flags
------------
  --version   Print version and exit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Terminal helpers (no deps — pure ANSI, graceful fallback on no-TTY / Windows)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or (
    os.name == "nt" and os.environ.get("TERM") not in (None, "")
)

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def bold(t: str)    -> str: return _c("1", t)
def green(t: str)   -> str: return _c("32", t)
def yellow(t: str)  -> str: return _c("33", t)
def cyan(t: str)    -> str: return _c("36", t)
def dim(t: str)     -> str: return _c("2", t)
def red(t: str)     -> str: return _c("31", t)

def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return green("█" * filled) + dim("░" * (width - filled))

def _score_color(score: float) -> str:
    if score >= 0.7:   return green(f"{score:.4f}")
    if score >= 0.35:  return yellow(f"{score:.4f}")
    return dim(f"{score:.4f}")


# ---------------------------------------------------------------------------
# Progress helper (tqdm if available, else simple counter)
# ---------------------------------------------------------------------------

def _iter_progress(items: list, desc: str = ""):
    try:
        from tqdm import tqdm
        yield from tqdm(items, desc=desc, unit="file", ncols=72)
    except ImportError:
        total = len(items)
        for i, item in enumerate(items, 1):
            print(f"\r{desc} [{i}/{total}]", end="", flush=True)
            yield item
        if total:
            print()


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def _cmd_predict(args: argparse.Namespace) -> None:
    from audiotimm import Classifier

    clf = Classifier.load(args.model or None, device=args.device)
    files: List[str] = args.files
    single = len(files) == 1
    results_out = []

    iterator = files if single else _iter_progress(files, desc="Classifying")

    for path in iterator:
        try:
            result = clf.predict(path)
        except Exception as exc:
            print(f"{red('ERROR')} {path}: {exc}", file=sys.stderr)
            continue

        rows = (
            result.above(args.threshold)
            if args.threshold is not None
            else result.top(args.top)
        )

        if args.json:
            results_out.append(result.as_dict())
        else:
            if not single:
                print(f"\n{bold(path)}")
            for label, score in rows:
                print(f"  {_score_color(score)}  {_bar(score)}  {label}")

    # JSON output
    if args.json:
        payload = results_out[0] if single else results_out
        text = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Saved to {args.output}")
        else:
            print(text)
    elif args.output:
        # Save as JSONL
        out = Path(args.output)
        with out.open("w", encoding="utf-8") as fh:
            for r in results_out:
                fh.write(json.dumps(r) + "\n")
        print(f"Saved {len(results_out)} results to {out}")


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

def _cmd_embed(args: argparse.Namespace) -> None:
    from audiotimm import Classifier

    clf = Classifier.load(args.model or None, device=args.device)
    files: List[str] = args.files
    single = len(files) == 1

    embeddings: dict[str, np.ndarray] = {}
    iterator = files if single else _iter_progress(files, desc="Embedding")

    for path in iterator:
        try:
            emb = clf.embed(path)
            embeddings[path] = emb
        except Exception as exc:
            print(f"{red('ERROR')} {path}: {exc}", file=sys.stderr)

    if not embeddings:
        sys.exit(1)

    if args.output:
        out = Path(args.output)
        fmt = args.format or ("npy" if single else "npz")
        if fmt == "npy" and single:
            np.save(str(out), list(embeddings.values())[0])
            print(f"Saved embedding {list(embeddings.values())[0].shape} → {out}")
        elif fmt == "npz":
            np.savez(str(out), **{Path(k).stem: v for k, v in embeddings.items()})
            print(f"Saved {len(embeddings)} embeddings → {out}")
        elif fmt == "csv":
            import csv
            with out.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["file"] + [f"dim_{i}" for i in range(list(embeddings.values())[0].shape[0])])
                for path_, emb in embeddings.items():
                    writer.writerow([path_] + emb.tolist())
            print(f"Saved {len(embeddings)} embeddings → {out}")
        else:
            print(f"{red('Unknown format')} {fmt!r}. Use npy, npz, or csv.")
            sys.exit(1)
    else:
        for path_, emb in embeddings.items():
            print(f"{bold(path_)}")
            print(f"  shape : {emb.shape}")
            print(f"  dtype : {emb.dtype}")
            print(f"  norm  : {float(np.linalg.norm(emb)):.4f}")
            print(f"  mean  : {float(emb.mean()):.4f}  std: {float(emb.std()):.4f}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def _cmd_list(args: argparse.Namespace) -> None:
    from audiotimm import registry

    specs = registry.list(
        wave=args.wave or None,
        task=args.task or None,
    )
    if args.family:
        specs = [s for s in specs if s.family == args.family]
    if not specs:
        print("No models match the given filters.")
        return

    if args.json:
        out = [
            {
                "name": s.name, "family": s.family, "wave": s.wave,
                "sample_rate": s.sample_rate, "n_classes": s.n_classes,
                "embed_dim": s.embed_dim, "task": s.task,
                "extra": s.extra, "description": s.description,
            }
            for s in sorted(specs, key=lambda x: (x.wave, x.name))
        ]
        print(json.dumps(out, indent=2))
        return

    # Group by wave
    from itertools import groupby
    sorted_specs = sorted(specs, key=lambda x: (x.wave, x.family, x.name))

    col_name  = 32
    col_fam   = 10
    col_wave  =  4
    col_sr    =  7
    col_cls   =  7
    sep = "─"

    header = (
        f"  {bold('Name'):<{col_name+8}}  "
        f"{bold('Family'):<{col_fam+8}}  "
        f"{bold('Wave'):<{col_wave+8}}  "
        f"{bold('SR'):>{col_sr+8}}  "
        f"{bold('Classes'):>{col_cls+8}}  "
        f"{bold('Description')}"
    )

    print()
    for wave, group in groupby(sorted_specs, key=lambda x: x.wave):
        group = list(group)
        print(f"  {cyan(bold(f'Wave {wave}'))}  {dim(f'— {len(group)} model(s)')}")
        print(f"  {sep * 90}")
        for s in group:
            extra_tag = dim(f"  [{s.extra}]") if s.extra else ""
            print(
                f"  {yellow(s.name):<{col_name + 9}}  "
                f"{s.family:<{col_fam}}  "
                f"{s.wave:<{col_wave}}  "
                f"{s.sample_rate:>{col_sr}}  "
                f"{str(s.n_classes) if s.n_classes else '—':>{col_cls}}  "
                f"{dim(s.description[:60])}"
                f"{extra_tag}"
            )
        print()


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def _cmd_info(args: argparse.Namespace) -> None:
    from audiotimm import registry

    try:
        s = registry.get(args.model)
    except ValueError as exc:
        print(red(str(exc)), file=sys.stderr)
        sys.exit(1)

    print()
    print(f"  {bold(s.name)}")
    print(f"  {'─' * 60}")
    rows = [
        ("Family",      s.family),
        ("Wave",        s.wave),
        ("Task",        s.task),
        ("Sample rate", f"{s.sample_rate} Hz"),
        ("Classes",     str(s.n_classes) if s.n_classes else "—"),
        ("Embed dim",   str(s.embed_dim)),
        ("Extras",      f"pip install audiotimm[{s.extra}]" if s.extra else "core (no extras needed)"),
        ("Checkpoint",  s.checkpoint),
        ("Description", s.description),
    ]
    for key, val in rows:
        print(f"  {cyan(key + ':'):<28} {val}")
    print()

    print(f"  {bold('Usage:')}")
    print(f"    {dim('from audiotimm import Classifier')}")
    print(f"    {dim(f'clf = Classifier.load({s.name!r})')}")
    print(f"    {dim('result = clf.predict(\"audio.wav\")')}")
    if s.task == "embed":
        print(f"    {dim('emb = clf.embed(\"audio.wav\")  # shape (' + str(s.embed_dim) + ',)')}")
    print()


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def _cmd_benchmark(args: argparse.Namespace) -> None:
    from audiotimm import Classifier

    clf = Classifier.load(args.model or None, device=args.device)
    path = args.file
    runs = args.runs

    # warm-up
    print(f"Warming up {bold(clf.model_name)} on {path} …")
    clf.predict(path)

    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        clf.predict(path)
        times.append(time.perf_counter() - t0)

    arr = np.array(times) * 1000  # ms
    print(f"\n  {bold('Benchmark:')} {clf.model_name}  ×{runs} runs")
    print(f"  {'─' * 40}")
    print(f"  mean   : {arr.mean():.1f} ms")
    print(f"  median : {np.median(arr):.1f} ms")
    print(f"  min    : {arr.min():.1f} ms")
    print(f"  max    : {arr.max():.1f} ms")
    print(f"  std    : {arr.std():.1f} ms")
    print()


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audiotimm",
        description=bold("audiotimm — The Model Hub for Audio Intelligence"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dim(
            "Examples:\n"
            "  audiotimm predict dog.wav\n"
            "  audiotimm predict *.wav --model ast-10-10 --top 3\n"
            "  audiotimm predict dog.wav --threshold 0.3 --json\n"
            "  audiotimm embed dog.wav --output emb.npy\n"
            "  audiotimm embed *.wav --output embeddings.npz\n"
            "  audiotimm list --wave M1\n"
            "  audiotimm info beats-iter3plus-as2m-cpt2\n"
            "  audiotimm benchmark dog.wav --model panns-cnn14 --runs 20\n"
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {_get_version()}",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── predict ─────────────────────────────────────────────────────────────
    p = sub.add_parser("predict", help="Classify one or more audio files")
    p.add_argument("files", nargs="+", metavar="FILE", help="Audio file path(s) — supports glob patterns")
    p.add_argument("-m", "--model",     default=None,  help="Model zoo-id  (default: panns-cnn14)")
    p.add_argument("-k", "--top",       type=int, default=5, metavar="K", help="Top-k results  (default: 5)")
    p.add_argument("-t", "--threshold", type=float, default=None, metavar="T",
                   help="Show all labels with score ≥ T  (overrides --top)")
    p.add_argument("-o", "--output",    default=None, metavar="FILE", help="Save results to FILE as JSON/JSONL")
    p.add_argument("--device",          default="cpu", help="Torch device  (default: cpu)")
    p.add_argument("--json",            action="store_true", help="Print results as JSON")
    p.set_defaults(func=_cmd_predict)

    # ── embed ────────────────────────────────────────────────────────────────
    e = sub.add_parser("embed", help="Extract audio embeddings")
    e.add_argument("files", nargs="+", metavar="FILE")
    e.add_argument("-m", "--model",   default=None)
    e.add_argument("-o", "--output",  default=None, metavar="FILE",
                   help=".npy (single), .npz (batch), or .csv")
    e.add_argument("--format",        choices=["npy", "npz", "csv"], default=None,
                   help="Output format  (auto-detected from --output extension)")
    e.add_argument("--device",        default="cpu")
    e.set_defaults(func=_cmd_embed)

    # ── list ─────────────────────────────────────────────────────────────────
    ls = sub.add_parser("list", help="List available models")
    ls.add_argument("--wave",   default=None, help="Filter by wave  (M0, M1, M2, …)")
    ls.add_argument("--task",   default=None, help="Filter by task  (tagging, zero-shot, embed, asr)")
    ls.add_argument("--family", default=None, help="Filter by family  (panns, ast, beats, htsat, …)")
    ls.add_argument("--json",   action="store_true", help="Output as JSON array")
    ls.set_defaults(func=_cmd_list)

    # ── info ─────────────────────────────────────────────────────────────────
    inf = sub.add_parser("info", help="Show detailed info for a model")
    inf.add_argument("model", help="Model zoo-id")
    inf.set_defaults(func=_cmd_info)

    # ── benchmark ────────────────────────────────────────────────────────────
    bm = sub.add_parser("benchmark", help="Time inference for a model")
    bm.add_argument("file",            metavar="FILE",   help="Audio file to benchmark on")
    bm.add_argument("-m", "--model",   default=None,     help="Model zoo-id  (default: panns-cnn14)")
    bm.add_argument("-n", "--runs",    type=int, default=10, help="Number of timed runs  (default: 10)")
    bm.add_argument("--device",        default="cpu")
    bm.set_defaults(func=_cmd_benchmark)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, FileNotFoundError) as exc:
        print(f"{red('Error:')} {exc}", file=sys.stderr)
        sys.exit(1)


def _get_version() -> str:
    try:
        from audiotimm import __version__
        return __version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
