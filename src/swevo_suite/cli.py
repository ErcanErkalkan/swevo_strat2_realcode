from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import fields
from pathlib import Path

from .manifest import load_manifest, write_run_matrix
from .paths import GENERATED, ensure_generated_dirs
from .schemas import RunResult
from .smoke import synthesize_result
from .comparators import REGISTRY

ABLATION_METHODS = {"A1_NoSeed", "A2_NoJDE", "A3_NoLNS"}


def _build_comparator(method_id: str):
    cls = REGISTRY[method_id]
    return cls(method_id) if method_id in ABLATION_METHODS else cls()


def cmd_plan(args: argparse.Namespace) -> None:
    rows = load_manifest(Path(args.manifest) if args.manifest else None)
    out = write_run_matrix(rows, Path(args.output) if args.output else None)
    print(out)


def cmd_run(args: argparse.Namespace) -> None:
    ensure_generated_dirs()
    rows = load_manifest(Path(args.manifest) if args.manifest else None)
    if args.limit:
        rows = rows[: args.limit]
    if args.method:
        rows = [r for r in rows if r.method_id == args.method]
    if args.instance:
        rows = [r for r in rows if r.instance_id == args.instance]
    out = Path(args.output) if args.output else (GENERATED / "master_runs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    fieldnames = [f.name for f in fields(RunResult)]
    total = len(rows)
    completed = 0
    failures = 0
    with out.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if args.overwrite or not out.exists() or out.stat().st_size == 0:
            writer.writeheader()
            f.flush()
        for idx, plan in enumerate(rows, start=1):
            if args.budget_override is not None:
                plan.eval_budget = int(args.budget_override)
            try:
                if args.smoke:
                    result = synthesize_result(plan)
                else:
                    comparator = _build_comparator(plan.method_id)
                    result = comparator.run(plan)
                writer.writerow(result.to_dict())
                f.flush()
                completed += 1
                if args.progress:
                    print(f"[{idx}/{total}] OK {plan.run_id}", file=sys.stderr)
            except Exception as exc:  # pragma: no cover - operational path
                failures += 1
                if args.progress:
                    print(f"[{idx}/{total}] FAIL {plan.run_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    raise
    print(f"completed={completed} failures={failures} output={out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--manifest")
    plan.add_argument("--output")
    plan.set_defaults(func=cmd_plan)
    run = sub.add_parser("run")
    run.add_argument("--manifest")
    run.add_argument("--output")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--method")
    run.add_argument("--instance")
    run.add_argument("--limit", type=int)
    run.add_argument("--budget-override", type=int)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--progress", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.set_defaults(func=cmd_run)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
