#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated_real"
BATCH_DIR = ROOT / "configs" / "batches"

METHODS = [
    "EDE",
    "StdDE",
    "ALNS_MS",
    "HGS_MS",
    "ILS_MS",
    "A1_NoSeed",
    "A2_NoJDE",
    "A3_NoLNS",
]


def main() -> int:
    GENERATED.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SWEVO_REQUIRE_REAL_BENCHMARKS"] = "1"
    env["PYTHONPATH"] = "src"

    procs: list[tuple[str, subprocess.Popen, Path]] = []
    for method in METHODS:
        manifest = BATCH_DIR / f"manifest_{method}.csv"
        output = GENERATED / f"batch_{method}.csv"
        log_path = GENERATED / f"batch_{method}.log"

        if not manifest.exists():
            print(f"Missing manifest for {method}: {manifest}")
            return 1

        if log_path.exists():
            log_path.unlink()

        log_file = log_path.open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_manifest.py"),
            "run",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--overwrite",
            "--progress",
        ]
        print(f"Starting {method} -> {output.name}")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((method, proc, log_file))

    print(f"Started {len(procs)} parallel batch processes.")
    sys.stdout.flush()

    failures = 0
    while procs:
        time.sleep(30)
        for method, proc, log_file in list(procs):
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                procs.remove((method, proc, log_file))
                print(f"Batch {method} finished with returncode={ret}")
                if ret != 0:
                    failures += 1
                    print(f"ERROR in batch {method}. Last log lines:")
                    with (GENERATED / f"batch_{method}.log").open("r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()[-20:]
                        for line in lines:
                            print(line.rstrip())
                else:
                    print(f"Batch {method} completed successfully.")
                print(f"{len(procs)} batches remaining")
                sys.stdout.flush()

    print("All batches finished.")
    if failures:
        print(f"{failures} batches failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
