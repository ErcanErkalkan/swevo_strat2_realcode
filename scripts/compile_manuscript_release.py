#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
TEX = MANUSCRIPT / "elsarticle-template-num.tex"
BIB = MANUSCRIPT / "cas-refs.bib"
BST = MANUSCRIPT / "elsarticle-num.bst"
BUILD_PDF = MANUSCRIPT / "elsarticle-template-num.pdf"
PDF = MANUSCRIPT / "manuscript.pdf"
PROVENANCE = MANUSCRIPT / "build_provenance.json"
# Fixed PDF metadata epoch for bit-reproducible clean builds; not an experiment timestamp.
SOURCE_DATE_EPOCH = "1788436800"  # 2026-09-03T12:00:00Z
COMPILE_COMMAND = [
    "latexmk",
    "-pdf",
    "-bibtex",
    "-interaction=nonstopmode",
    "-halt-on-error",
    TEX.name,
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_pages(path: Path) -> int:
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages)
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    raise RuntimeError("Could not determine manuscript PDF page count (need pypdf or pdfinfo)")


def expected_record() -> dict[str, object]:
    return {
        "schema": "swevo-manuscript-build-provenance-v3",
        "tex_file": TEX.name,
        "tex_sha256": sha256(TEX),
        "bib_file": BIB.name,
        "bib_sha256": sha256(BIB),
        "bst_file": BST.name,
        "bst_sha256": sha256(BST),
        "pdf_file": PDF.name,
        "pdf_sha256": sha256(PDF),
        "pages": pdf_pages(PDF),
        "build_command": " ".join(COMPILE_COMMAND),
        "source_date_epoch": int(SOURCE_DATE_EPOCH),
        "force_source_date": True,
    }


def verify() -> int:
    required = [TEX, BIB, BST, PDF, PROVENANCE]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        print("Missing manuscript release files:", ", ".join(missing), file=sys.stderr)
        return 1
    try:
        stored = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Invalid {PROVENANCE}: {exc}", file=sys.stderr)
        return 1
    current = expected_record()
    mismatches = []
    for key in ("schema", "tex_file", "tex_sha256", "bib_file", "bib_sha256", "bst_file", "bst_sha256", "pdf_file", "pdf_sha256", "pages", "build_command", "source_date_epoch", "force_source_date"):
        if stored.get(key) != current.get(key):
            mismatches.append(f"{key}: stored={stored.get(key)!r}, current={current.get(key)!r}")
    if mismatches:
        print("Manuscript build provenance mismatch:", file=sys.stderr)
        for item in mismatches:
            print(" -", item, file=sys.stderr)
        return 1
    print(
        f"manuscript_provenance=PASS pages={current['pages']} "
        f"pdf_sha256={str(current['pdf_sha256'])}"
    )
    return 0


def build(clean: bool) -> int:
    missing_sources = [p.name for p in (TEX, BIB, BST) if not p.exists()]
    if missing_sources:
        print("Missing manuscript source/build dependency: " + ", ".join(missing_sources), file=sys.stderr)
        return 1
    if shutil.which("latexmk") is None:
        print("latexmk is required to compile the release manuscript", file=sys.stderr)
        return 1
    if shutil.which("bibtex") is None:
        print("bibtex is required to compile the release manuscript bibliography", file=sys.stderr)
        return 1
    build_env = __import__("os").environ.copy()
    build_env["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    build_env["FORCE_SOURCE_DATE"] = "1"
    if clean:
        subprocess.run(["latexmk", "-C", TEX.name], cwd=MANUSCRIPT, env=build_env, check=False)
    print("Running:", " ".join(COMPILE_COMMAND), flush=True)
    subprocess.run(COMPILE_COMMAND, cwd=MANUSCRIPT, env=build_env, check=True)
    if not BUILD_PDF.exists():
        print(f"Expected compiled PDF missing: {BUILD_PDF}", file=sys.stderr)
        return 1
    shutil.copy2(BUILD_PDF, PDF)
    record = expected_record()
    record["built_utc"] = datetime.now(timezone.utc).isoformat()
    record["python"] = sys.version
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"compiled={PDF} pages={record['pages']} pdf_sha256={record['pdf_sha256']}"
    )
    return verify()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile or verify the release manuscript and its SHA-256 build provenance.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--build", action="store_true")
    ap.add_argument("--clean", action="store_true", help="Run latexmk -C before compiling")
    args = ap.parse_args()
    if args.verify_only:
        return verify()
    return build(args.clean)


if __name__ == "__main__":
    raise SystemExit(main())
