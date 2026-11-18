#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "experiment_manifest_submission_primary.csv"
BENCH_ROOT = ROOT / "data" / "benchmarks"

SOURCES = {
    ("solomon", 100): "https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip",
    ("hg", 200): "https://www.sintef.no/globalassets/project/top/vrptw/homberger/200/homberger_200_customer_instances.zip",
    ("hg", 400): "https://www.sintef.no/globalassets/project/top/vrptw/homberger/400/homberger_400_customer_instances.zip",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def primary_instances(manifest: Path) -> dict[tuple[str, int], list[str]]:
    groups: dict[tuple[str, int], set[str]] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            family = str(row["benchmark_family"]).strip().lower()
            n = int(float(row["customer_count"]))
            key = (family, n)
            if key not in SOURCES:
                raise SystemExit(f"No official acquisition source registered for primary group {key}")
            groups.setdefault(key, set()).add(str(row["instance_id"]).strip().upper())
    return {key: sorted(values) for key, values in sorted(groups.items())}


def target_dir(key: tuple[str, int]) -> Path:
    family, _ = key
    return BENCH_ROOT / ("solomon" if family == "solomon" else "homberger")


def find_local_instance(key: tuple[str, int], instance_id: str) -> Path | None:
    stem = instance_id.lower()
    family_dir = target_dir(key)
    if not family_dir.exists():
        return None
    matches = sorted(
        p for p in family_dir.rglob("*")
        if p.is_file() and p.stem.lower() == stem and p.suffix.lower() in {".txt", ".vrp", ".csv"}
    )
    return matches[0] if matches else None


def verify(groups: dict[tuple[str, int], list[str]]) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for key, ids in groups.items():
        for instance_id in ids:
            path = find_local_instance(key, instance_id)
            if path is None:
                missing.append(instance_id)
                continue
            rows.append({
                "benchmark_family": key[0],
                "customer_count": key[1],
                "instance_id": instance_id,
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "official_source_archive": SOURCES[key],
            })
    return rows, missing


def download_archive(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SWEVO reproducibility acquisition/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
    if not data.startswith(b"PK"):
        raise RuntimeError(f"Downloaded response from {url} is not a ZIP archive")
    return data


def install_group(key: tuple[str, int], ids: list[str], archive: bytes) -> None:
    wanted = {instance_id.lower(): instance_id for instance_id in ids}
    found: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            stem = Path(info.filename).stem.lower()
            if stem not in wanted:
                continue
            if stem in found:
                raise RuntimeError(f"Duplicate {wanted[stem]} in official archive for {key}")
            found[stem] = zf.read(info)
    missing = [wanted[stem] for stem in wanted if stem not in found]
    if missing:
        raise RuntimeError(f"Official archive for {key} is missing expected primary instances: {missing}")
    out_dir = target_dir(key)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, data in found.items():
        (out_dir / f"{wanted[stem]}.txt").write_bytes(data)


def write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "benchmark_family", "customer_count", "instance_id", "relative_path",
        "bytes", "sha256", "official_source_archive",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (str(r["benchmark_family"]), int(r["customer_count"]), str(r["instance_id"]))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Acquire/verify the 36 public benchmark instances used by the submission-primary manifest.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--inventory", default=str(ROOT / "configs" / "benchmark_inventory_submission_primary.csv"))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    inventory = Path(args.inventory)
    groups = primary_instances(manifest)
    expected = sum(len(ids) for ids in groups.values())
    if expected != 36:
        raise SystemExit(f"Expected 36 unique primary instances; manifest resolves to {expected}")

    rows, missing = verify(groups)
    if args.verify_only:
        if missing:
            print("Missing real benchmark instances:", ", ".join(sorted(missing)))
            return 1
        write_inventory(inventory, rows)
        print(f"Verified {len(rows)} real benchmark files; wrote {inventory}")
        return 0

    if missing:
        missing_set = set(missing)
        for key, ids in groups.items():
            if not any(i in missing_set for i in ids):
                continue
            url = SOURCES[key]
            print(f"Downloading official archive: {url}", flush=True)
            archive = download_archive(url)
            print(f"archive_sha256={sha256_bytes(archive)} bytes={len(archive)}", flush=True)
            install_group(key, ids, archive)

    rows, missing = verify(groups)
    if missing:
        raise SystemExit("Benchmark acquisition incomplete: " + ", ".join(sorted(missing)))
    write_inventory(inventory, rows)
    print(f"Acquired/verified {len(rows)} real benchmark files; wrote {inventory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
