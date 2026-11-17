#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
SOURCE = MANUSCRIPT / "graphical_abstract_source.tex"
BUILD_PDF = MANUSCRIPT / "graphical_abstract_source.pdf"
PDF = MANUSCRIPT / "graphical_abstract.pdf"
PNG = MANUSCRIPT / "graphical_abstract.png"
PROVENANCE = MANUSCRIPT / "graphical_abstract_provenance.json"
SOURCE_DATE_EPOCH = "1788436800"
DPI = 220
COMPILE_COMMAND = [
    "latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", SOURCE.name,
]
RASTER_COMMAND = [
    "pdftoppm", "-png", "-r", str(DPI), "-singlefile", PDF.name, "graphical_abstract",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise RuntimeError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", data[16:24])


def expected_record() -> dict[str, object]:
    width, height = png_dimensions(PNG)
    return {
        "schema": "swevo-graphical-abstract-provenance-v1",
        "source_file": SOURCE.name,
        "source_sha256": sha256(SOURCE),
        "pdf_file": PDF.name,
        "pdf_sha256": sha256(PDF),
        "png_file": PNG.name,
        "png_sha256": sha256(PNG),
        "png_width_px": width,
        "png_height_px": height,
        "raster_dpi": DPI,
        "compile_command": " ".join(COMPILE_COMMAND),
        "raster_command": " ".join(RASTER_COMMAND),
        "source_date_epoch": int(SOURCE_DATE_EPOCH),
        "force_source_date": True,
    }


def verify() -> int:
    required = [SOURCE, PDF, PNG, PROVENANCE]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        print("Missing graphical-abstract release files: " + ", ".join(missing), file=sys.stderr)
        return 1
    try:
        stored = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Invalid {PROVENANCE}: {exc}", file=sys.stderr)
        return 1
    current = expected_record()
    mismatches = [
        f"{key}: stored={stored.get(key)!r}, current={value!r}"
        for key, value in current.items()
        if stored.get(key) != value
    ]
    if mismatches:
        print("Graphical-abstract provenance mismatch:", file=sys.stderr)
        for item in mismatches:
            print(" -", item, file=sys.stderr)
        return 1
    print(
        "graphical_abstract_provenance=PASS "
        f"pdf_sha256={current['pdf_sha256']} png_sha256={current['png_sha256']}"
    )
    return 0


def build(clean: bool) -> int:
    if not SOURCE.exists():
        print(f"Missing graphical abstract source: {SOURCE}", file=sys.stderr)
        return 1
    for exe in ("latexmk", "pdftoppm"):
        if shutil.which(exe) is None:
            print(f"{exe} is required for graphical-abstract release build", file=sys.stderr)
            return 1
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    env["FORCE_SOURCE_DATE"] = "1"
    if clean:
        subprocess.run(["latexmk", "-C", SOURCE.name], cwd=MANUSCRIPT, env=env, check=False)
    subprocess.run(COMPILE_COMMAND, cwd=MANUSCRIPT, env=env, check=True)
    if not BUILD_PDF.exists():
        print(f"Expected build PDF missing: {BUILD_PDF}", file=sys.stderr)
        return 1
    shutil.copy2(BUILD_PDF, PDF)
    subprocess.run(RASTER_COMMAND, cwd=MANUSCRIPT, env=env, check=True)
    if not PNG.exists():
        print(f"Expected raster PNG missing: {PNG}", file=sys.stderr)
        return 1
    PROVENANCE.write_text(json.dumps(expected_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verify()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build or verify deterministic graphical-abstract release artifacts.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--build", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    if args.verify_only:
        return verify()
    return build(args.clean)


if __name__ == "__main__":
    raise SystemExit(main())
