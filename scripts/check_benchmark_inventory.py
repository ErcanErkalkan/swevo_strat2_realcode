#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import csv

from swevo_suite.benchmark import InvalidBenchmarkInstanceId, _candidate_instance_paths, build_problem
from swevo_suite.paths import CONFIGS, GENERATED
from swevo_suite.schemas import RunPlan


def _infer_structure_class(instance_id: str) -> str:
    up = instance_id.upper()
    if up.startswith("RC"):
        return "mixed"
    if up.startswith("R"):
        return "random"
    if up.startswith("C"):
        return "clustered"
    return "mixed"


def _inventory_plan(row: dict[str, str]) -> RunPlan:
    structure_class = row.get("structure_class") or _infer_structure_class(row["instance_id"])
    return RunPlan(
        run_id=f"{row['instance_id']}__inventory__EDE__seed01",
        phase="inventory",
        benchmark_family=row["benchmark_family"],
        customer_count=int(row["customer_count"]),
        instance_id=row["instance_id"],
        structure_class=structure_class,
        scenario_id="S1_balanced",
        method_id="EDE",
        method_group="audit",
        ablation_flag="",
        seed=1,
        tier=row.get("default_tier", "small"),
        eval_budget=1,
        walltime_cap_s=1,
        status="planned",
    )


def _resolve_inventory_row(row: dict[str, str]) -> dict[str, object]:
    plan = _inventory_plan(row)
    matches = _candidate_instance_paths(plan)
    status = "ok"
    resolved_path = ""
    error = ""
    try:
        problem = build_problem(plan, require_real=True)
        resolved_path = problem.source_path
    except Exception as exc:  # pragma: no cover - operational path
        if isinstance(exc, InvalidBenchmarkInstanceId):
            status = "invalid_public_instance_id"
        elif isinstance(exc, FileNotFoundError):
            status = "missing"
        else:
            status = "error"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "instance_id": row["instance_id"],
        "benchmark_family": row["benchmark_family"],
        "customer_count": int(row["customer_count"]),
        "default_tier": row.get("default_tier", ""),
        "match_count": len(matches),
        "status": status,
        "resolved_path": resolved_path,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=str(CONFIGS / "benchmark_inventory.csv"))
    parser.add_argument("--output", default=str(GENERATED / "benchmark_inventory_resolution.csv"))
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    output_path = Path(args.output)

    with inventory_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    resolved_rows = [_resolve_inventory_row(row) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id",
                "benchmark_family",
                "customer_count",
                "default_tier",
                "match_count",
                "status",
                "resolved_path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(resolved_rows)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
