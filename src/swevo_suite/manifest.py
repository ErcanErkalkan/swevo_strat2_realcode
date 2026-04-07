from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, List, Optional
from .paths import CONFIGS, GENERATED
from .schemas import RunPlan

def load_manifest(path: Optional[Path] = None) -> List[RunPlan]:
    manifest_path = path or (CONFIGS / "experiment_manifest_full.csv")
    rows: List[RunPlan] = []
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = "__".join([
                row["instance_id"], row["scenario_id"], row["method_id"], f"seed{int(row['seed']):02d}"
            ])
            rows.append(RunPlan(
                run_id=run_id,
                phase=row["phase"],
                benchmark_family=row["benchmark_family"],
                customer_count=int(row["customer_count"]),
                instance_id=row["instance_id"],
                structure_class=row["structure_class"],
                scenario_id=row["scenario_id"],
                method_id=row["method_id"],
                method_group=row["method_group"],
                ablation_flag=row["ablation_flag"],
                seed=int(row["seed"]),
                tier=row["tier"],
                eval_budget=int(row["eval_budget"]),
                walltime_cap_s=int(row["walltime_cap_s"]),
                status=row.get("status", "planned"),
            ))
    return rows

def write_run_matrix(rows: Iterable[RunPlan], path: Optional[Path] = None) -> Path:
    out = path or (GENERATED / "run_matrix_with_ids.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fieldnames = list(RunPlan.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return out
