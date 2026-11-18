# Reproduction

Use Python >=3.11; the verified environment is CPython 3.14.3 on Windows 11 build 26200. Install `requirements.lock.txt` into a fresh virtual environment, then run from the extracted reviewer supplement root:

```
python -m pip install -r requirements.lock.txt
python -m pip check
python -m pytest -q -p no:cacheprovider
python scripts/finalize.py --core
```

This reproduces 41 outputs in `results/primary/` from the locked ledger, configurations, secondary data, templates and frozen reference. `results/primary/provenance.json` records input roles and output hashes with disjoint input/output sets. The core does not require benchmark downloads, manuscript assets, an historical archive or new solver execution. Compare outputs with `scripts/compare_outputs.py`; exact byte equality and numerical equivalence are reported separately. Ordinary numeric tolerance is rtol=1e-10, atol=1e-12; p-value absolute tolerance is 1e-300. Columns, text labels, row counts and missingness must also match.

The supplement contains a scientific ablation status and the unmodified preregistration, without operational logs or raw partial checkpoints. Re-auditing actual partial records requires the full local project: `python scripts/audit_ablation.py --campaign data/ablation --out results/ablation`. The reported 48/1080 status is a snapshot, not a completed experiment. See `ABLATION_CONTINUATION.md` for the archive-backed locked execution context, which is intentionally not required by core reproduction.

For the article, extract `manuscript_source.zip` into a new directory. In its `manuscript/` subdirectory run pdflatex, bibtex, pdflatex, pdflatex on `main`, then pdflatex on `graphical_abstract`. The sibling `results/primary/` directory contains all five required generated TeX inputs. MiKTeX 25.4 and Poppler 24.04 were used. No `siunitx` dependency is needed. Portal validation is `python scripts/finalize.py --portal` from the extracted portal root with the declared Python dependencies installed.

Benchmark acquisition metadata and SHA-256 values are in `configs/benchmark_inventory_submission_primary.csv`; `scripts/acquire_primary_benchmarks.py` downloads and verifies official files if optimizer execution is needed. Benchmark bytes are not redistributed in reviewer/public snapshots. The public code snapshot excludes the primary raw ledger, secondary raw outcomes and primary run manifest; core reproduction requires the complete reviewer supplement. Code-only tests run without those raw inputs.
