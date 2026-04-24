#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import csv
import re
from collections import Counter

from swevo_suite.benchmark import KNOWN_PUBLIC_INSTANCE_IDS, build_problem
from swevo_suite.paths import CONFIGS, GENERATED
from swevo_suite.schemas import RunPlan


INSTANCE_ID_RE = re.compile(r"^([A-Za-z_]+?)(\d+)$")


def _instance_parts(instance_id: str) -> tuple[str, int | None]:
    match = INSTANCE_ID_RE.match(instance_id.strip())
    if not match:
        return instance_id.upper(), None
    prefix, numeric = match.groups()
    return prefix.upper(), int(numeric)


def _same_bucket_candidates(
    row: dict[str, str],
    used_ids: set[str],
) -> list[str]:
    family = row["benchmark_family"].lower()
    valid_ids = KNOWN_PUBLIC_INSTANCE_IDS.get(family, frozenset())
    if not valid_ids:
        return []

    target_prefix, target_num = _instance_parts(row["instance_id"])
    target_hundreds = None if target_num is None else target_num // 100
    out: list[tuple[tuple[int, int, int, str], str]] = []
    for candidate in sorted(valid_ids):
        if candidate.upper() in used_ids:
            continue
        cand_prefix, cand_num = _instance_parts(candidate)
        if cand_prefix != target_prefix:
            continue
        if target_num is None or cand_num is None:
            score = (1, 1, 999999, candidate)
        else:
            score = (
                0 if cand_num // 100 == target_hundreds else 1,
                abs(cand_num - target_num),
                cand_num,
                candidate,
            )
        out.append((score, candidate))
    return [candidate for _, candidate in sorted(out)]


def _candidate_plan(row: dict[str, str], instance_id: str) -> RunPlan:
    return RunPlan(
        run_id=f"{instance_id}__repair__EDE__seed01",
        phase="repair",
        benchmark_family=row["benchmark_family"],
        customer_count=int(row["customer_count"]),
        instance_id=instance_id,
        structure_class=row["structure_class"],
        scenario_id="S1_balanced",
        method_id="EDE",
        method_group="audit",
        ablation_flag="",
        seed=1,
        tier=row["default_tier"],
        eval_budget=1,
        walltime_cap_s=1,
        status="planned",
    )


def _candidate_is_real(row: dict[str, str], instance_id: str) -> bool:
    try:
        build_problem(_candidate_plan(row, instance_id), require_real=True)
    except Exception:
        return False
    return True


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    elif path.name == "benchmark_repair_suggestions.csv":
        fieldnames = [
            "instance_id",
            "benchmark_family",
            "customer_count",
            "structure_class",
            "default_tier",
            "current_status",
            "blocked_manifest_rows",
            "suggestion_1",
            "suggestion_1_reason",
            "suggestion_2",
            "suggestion_2_reason",
            "suggestion_3",
            "suggestion_3_reason",
            "error",
        ]
    else:
        fieldnames = []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _replacement_reason(source_id: str, replacement_id: str) -> str:
    src_prefix, src_num = _instance_parts(source_id)
    dst_prefix, dst_num = _instance_parts(replacement_id)
    reasons = ["same_prefix" if src_prefix == dst_prefix else "different_prefix"]
    if src_num is not None and dst_num is not None and src_num // 100 == dst_num // 100:
        reasons.append("same_series")
    if src_num is not None and dst_num is not None:
        reasons.append(f"numeric_gap={abs(src_num - dst_num)}")
    return ",".join(reasons)


def build_repair_plan(
    inventory_rows: list[dict[str, str]],
    resolution_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    status_by_instance = {
        row["instance_id"].upper(): row
        for row in resolution_rows
    }
    used_ids = {row["instance_id"].upper() for row in inventory_rows}
    manifest_counts = Counter(row["instance_id"].upper() for row in manifest_rows)

    suggestions: list[dict[str, object]] = []
    replacement_map: dict[str, str] = {}

    for row in inventory_rows:
        instance_id = row["instance_id"].upper()
        resolution = status_by_instance.get(instance_id, {})
        status = resolution.get("status", "unknown")
        if status != "invalid_public_instance_id":
            continue

        candidates = [
            candidate
            for candidate in _same_bucket_candidates(row, used_ids)
            if _candidate_is_real(row, candidate)
        ]
        top = candidates[:3]
        if top:
            replacement_map[instance_id] = top[0]

        suggestions.append(
            {
                "instance_id": row["instance_id"],
                "benchmark_family": row["benchmark_family"],
                "customer_count": int(row["customer_count"]),
                "structure_class": row["structure_class"],
                "default_tier": row["default_tier"],
                "current_status": status,
                "blocked_manifest_rows": manifest_counts.get(instance_id, 0),
                "suggestion_1": top[0] if len(top) > 0 else "",
                "suggestion_1_reason": _replacement_reason(row["instance_id"], top[0]) if len(top) > 0 else "",
                "suggestion_2": top[1] if len(top) > 1 else "",
                "suggestion_2_reason": _replacement_reason(row["instance_id"], top[1]) if len(top) > 1 else "",
                "suggestion_3": top[2] if len(top) > 2 else "",
                "suggestion_3_reason": _replacement_reason(row["instance_id"], top[2]) if len(top) > 2 else "",
                "error": resolution.get("error", ""),
            }
        )

    return suggestions, replacement_map


def _apply_replacements(rows: list[dict[str, str]], replacement_map: dict[str, str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        replaced = dict(row)
        instance_id = row["instance_id"].upper()
        if instance_id in replacement_map:
            replaced["instance_id"] = replacement_map[instance_id]
        out.append(replaced)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=str(CONFIGS / "benchmark_inventory.csv"))
    parser.add_argument("--resolution", default=str(GENERATED / "benchmark_inventory_resolution.csv"))
    parser.add_argument("--manifest", default=str(CONFIGS / "experiment_manifest_full.csv"))
    parser.add_argument("--suggestions-output", default=str(GENERATED / "benchmark_repair_suggestions.csv"))
    parser.add_argument("--proposal-inventory-output", default=str(CONFIGS / "benchmark_inventory_repair_proposal.csv"))
    parser.add_argument("--proposal-manifest-output", default=str(CONFIGS / "experiment_manifest_full_repair_proposal.csv"))
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    resolution_path = Path(args.resolution)
    manifest_path = Path(args.manifest)
    suggestions_output = Path(args.suggestions_output)
    proposal_inventory_output = Path(args.proposal_inventory_output)
    proposal_manifest_output = Path(args.proposal_manifest_output)

    inventory_rows = _load_csv(inventory_path)
    resolution_rows = _load_csv(resolution_path)
    manifest_rows = _load_csv(manifest_path)

    suggestions, replacement_map = build_repair_plan(inventory_rows, resolution_rows, manifest_rows)

    _write_csv(suggestions_output, suggestions)
    if suggestions:
        _write_csv(proposal_inventory_output, _apply_replacements(inventory_rows, replacement_map))
        _write_csv(proposal_manifest_output, _apply_replacements(manifest_rows, replacement_map))
        print(f"Wrote {suggestions_output}")
        print(f"Wrote {proposal_inventory_output}")
        print(f"Wrote {proposal_manifest_output}")
    else:
        print(f"Wrote {suggestions_output}")
        print("No invalid benchmark ids found; no repair proposal written.")


if __name__ == "__main__":
    main()
