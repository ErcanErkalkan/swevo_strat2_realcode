#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import csv
from collections import defaultdict
from typing import Iterable

from swevo_suite.benchmark import build_problem
from swevo_suite.manifest import load_manifest
from swevo_suite.paths import CONFIGS, GENERATED


DEFAULT_METHODS = ("EDE", "ILS_MS", "StdDE")
DEFAULT_TIERS = ("small", "medium", "large")


def _load_manifest_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header: {path}")
        return list(reader.fieldnames), rows


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _chunked(values: list[int], size: int) -> list[list[int]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(CONFIGS / "experiment_manifest_full.csv"))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--tiers", nargs="+", default=list(DEFAULT_TIERS))
    parser.add_argument("--prefix", default="paper_main3_real")
    parser.add_argument("--seed-block-size", type=int, default=2)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    fieldnames, raw_rows = _load_manifest_rows(manifest_path)
    plans = load_manifest(manifest_path)
    if len(raw_rows) != len(plans):
        raise SystemExit(f"Manifest row mismatch in {manifest_path}")

    methods = tuple(dict.fromkeys(args.methods))
    tiers = tuple(dict.fromkeys(args.tiers))

    selected: list[tuple[dict[str, str], object]] = []
    for raw, plan in zip(raw_rows, plans):
        if raw["method_id"] not in methods:
            continue
        if raw["tier"] not in tiers:
            continue
        selected.append((raw, plan))

    if not selected:
        raise SystemExit("No rows selected from manifest")

    prefix = args.prefix
    requested_path = CONFIGS / f"experiment_manifest_{prefix}_requested.csv"
    ready_path = CONFIGS / f"experiment_manifest_{prefix}_ready.csv"
    blocked_path = CONFIGS / f"experiment_manifest_{prefix}_blocked.csv"
    batch_dir = CONFIGS / f"{prefix}_batches"
    resolution_path = GENERATED / f"{prefix}_preflight_resolution.csv"
    batch_index_path = GENERATED / f"{prefix}_batch_index.csv"
    report_path = GENERATED / f"{prefix}_preflight_report.txt"

    _write_csv(requested_path, fieldnames, (raw for raw, _ in selected))

    groups: dict[tuple[str, int, str, str, str], list[tuple[dict[str, str], object]]] = defaultdict(list)
    for raw, plan in selected:
        key = (
            plan.benchmark_family,
            plan.customer_count,
            plan.instance_id,
            plan.structure_class,
            plan.tier,
        )
        groups[key].append((raw, plan))

    resolution_rows: list[dict[str, object]] = []
    resolvable_keys: set[tuple[str, int, str, str, str]] = set()
    blocked_keys: set[tuple[str, int, str, str, str]] = set()
    for key in sorted(groups):
        plan = groups[key][0][1]
        status = "ok"
        source_kind = ""
        source_path = ""
        error = ""
        try:
            problem = build_problem(plan, require_real=True)
            source_kind = problem.source_kind
            source_path = problem.source_path
            if problem.source_kind != "real":
                status = "unexpected_source"
                error = f"Resolved non-real source_kind={problem.source_kind}"
                blocked_keys.add(key)
            else:
                resolvable_keys.add(key)
        except Exception as exc:  # pragma: no cover - operational path
            status = "missing" if isinstance(exc, FileNotFoundError) else "error"
            error = f"{type(exc).__name__}: {exc}"
            blocked_keys.add(key)

        resolution_rows.append(
            {
                "benchmark_family": plan.benchmark_family,
                "customer_count": plan.customer_count,
                "instance_id": plan.instance_id,
                "structure_class": plan.structure_class,
                "tier": plan.tier,
                "status": status,
                "source_kind": source_kind,
                "source_path": source_path,
                "affected_runs": len(groups[key]),
                "error": error,
            }
        )

    _write_csv(
        resolution_path,
        [
            "benchmark_family",
            "customer_count",
            "instance_id",
            "structure_class",
            "tier",
            "status",
            "source_kind",
            "source_path",
            "affected_runs",
            "error",
        ],
        resolution_rows,
    )

    ready_pairs = []
    blocked_pairs = []
    for raw, plan in selected:
        key = (
            plan.benchmark_family,
            plan.customer_count,
            plan.instance_id,
            plan.structure_class,
            plan.tier,
        )
        if key in resolvable_keys:
            ready_pairs.append((raw, plan))
        else:
            blocked_pairs.append((raw, plan))

    _write_csv(ready_path, fieldnames, (raw for raw, _ in ready_pairs))
    _write_csv(blocked_path, fieldnames, (raw for raw, _ in blocked_pairs))

    batch_rows: list[dict[str, object]] = []
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_counter = 0
    for tier in tiers:
        tier_pairs = [(raw, plan) for raw, plan in ready_pairs if plan.tier == tier]
        if not tier_pairs:
            continue
        seeds = sorted({plan.seed for _, plan in tier_pairs})
        for seed_block in _chunked(seeds, args.seed_block_size):
            batch_counter += 1
            seed_set = set(seed_block)
            batch_pairs = [(raw, plan) for raw, plan in tier_pairs if plan.seed in seed_set]
            batch_path = batch_dir / (
                f"experiment_manifest_{prefix}_batch{batch_counter:02d}_{tier}"
                f"_seeds_{seed_block[0]:02d}_{seed_block[-1]:02d}.csv"
            )
            _write_csv(batch_path, fieldnames, (raw for raw, _ in batch_pairs))
            batch_rows.append(
                {
                    "batch_id": f"batch{batch_counter:02d}",
                    "tier": tier,
                    "seed_start": seed_block[0],
                    "seed_end": seed_block[-1],
                    "run_count": len(batch_pairs),
                    "instance_count": len({plan.instance_id for _, plan in batch_pairs}),
                    "scenario_count": len({plan.scenario_id for _, plan in batch_pairs}),
                    "method_count": len({plan.method_id for _, plan in batch_pairs}),
                    "manifest_path": _rel(batch_path),
                }
            )

    _write_csv(
        batch_index_path,
        [
            "batch_id",
            "tier",
            "seed_start",
            "seed_end",
            "run_count",
            "instance_count",
            "scenario_count",
            "method_count",
            "manifest_path",
        ],
        batch_rows,
    )

    blocked_instances = sorted({plan.instance_id for _, plan in blocked_pairs})
    report_lines = [
        f"source_manifest={_rel(manifest_path)}",
        f"requested_manifest={_rel(requested_path)}",
        f"ready_manifest={_rel(ready_path)}",
        f"blocked_manifest={_rel(blocked_path)}",
        f"resolution_csv={_rel(resolution_path)}",
        f"batch_index_csv={_rel(batch_index_path)}",
        f"methods={','.join(methods)}",
        f"tiers={','.join(tiers)}",
        f"requested_runs={len(selected)}",
        f"ready_runs={len(ready_pairs)}",
        f"blocked_runs={len(blocked_pairs)}",
        f"blocked_instances={','.join(blocked_instances) if blocked_instances else 'none'}",
        f"batch_count={len(batch_rows)}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote {requested_path}")
    print(f"Wrote {ready_path}")
    print(f"Wrote {blocked_path}")
    print(f"Wrote {resolution_path}")
    print(f"Wrote {batch_index_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
