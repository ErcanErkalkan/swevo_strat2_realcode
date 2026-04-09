#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()

    sources = [Path(item) for item in args.src]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] | None = None
    seen_run_ids: set[str] = set()
    rows_written = 0
    with out.open("w", newline="") as f_out:
        writer = None
        for src in sources:
            with src.open(newline="") as f_in:
                reader = csv.DictReader(f_in)
                if reader.fieldnames is None:
                    raise SystemExit(f"Missing header in {src}")
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames)
                    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                    writer.writeheader()
                elif list(reader.fieldnames) != fieldnames:
                    raise SystemExit(f"Header mismatch in {src}")
                assert writer is not None
                for row in reader:
                    run_id = row.get("run_id", "")
                    if not args.no_dedupe and run_id:
                        if run_id in seen_run_ids:
                            continue
                        seen_run_ids.add(run_id)
                    writer.writerow(row)
                    rows_written += 1
    print(f"Wrote {out} rows={rows_written}")


if __name__ == "__main__":
    main()
