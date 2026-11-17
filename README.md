# E-DE for sustainability-aware VRPTW — C15 public code mirror

C15 adds a dependence sensitivity analysis, standalone core reproduction and
resumable ablation tooling to the frozen submission snapshot. The study compares
the integrated E-DE solver with four frozen internal implementations using
36 public VRPTW instances, three operating scenarios and 20 paired seeds
(10,800 runs). These are overlapping full-horizon operating regimes, not
sequential driver shifts. The study supports model-score solution quality,
not measured field savings, universal solver superiority or isolated component
effects. The confirmatory ablation is open and is not part of these conclusions.

## Run the public regression tests

Create a Python virtual environment, activate it, then run:

```sh
python -m pip install -r requirements-c15.lock.txt
python -m pytest -q -p no:cacheprovider
```

The frozen C15 package recorded CPython 3.14.3 (64-bit Windows). The public
checkout does not include the manuscript, so two manuscript-specific tests
skip. Run the full suite from the supplement to verify those checks as well.

## Reproduce the C15 evidence

The authoritative package is `EDE_SUSTAINABLE_VRPTW_C15_SUPPLEMENT.zip`,
SHA-256 `25906c2d2bc1c054911ef25bcfca8d6fba304755f8b9a238642274d57924e7a7`.
The public branch does not redistribute the manuscript, raw run ledgers,
confirmatory checkpoints or portal package. In a clean extraction of the
supplement, install the pinned dependencies and run:

```sh
python scripts/finalize_c15.py --core
python -m pytest -q -p no:cacheprovider
```

This generates the original 108 instance-by-scenario cell results and the
36-instance sensitivity, plus a 20,000-resample bootstrap that carries each
instance's three scenarios together. It does not rerun the optimization
campaign. The results are in [generated_submission](generated_submission),
with the estimands described in [the statistics addendum](docs/C15_STATS_ADDENDUM.md).
The core workflow requires neither the historical archive nor the cover letter.
The separately distributed portal package is verified with `--portal`.

To run the core command from this checkout instead of a clean supplement,
first stage the five pinned input files from that exact supplement at their
original relative paths, listed in [the evidence notes](generated_submission/README.md).
For ablation continuation, use the supplement and its locked protocol. The
code in this branch includes `scripts/run_c15_ablation.py` and
`scripts/audit_c15_ablation.py`, not the raw campaign contents.

Historical branches `submission-2026-09-02` and
`release/v0.2.0-submission-rc1` remain frozen. Their release documents describe
the older snapshot and should not be interpreted as C15 package metadata.
