# P8 submission freeze record

Manuscript: **Feasibility-First Enhanced Differential Evolution for Sustainability-Aware VRPTW with Shift-Indexed Operating Regimes**

Freeze date: 2026-09-02
Submission branch: `submission-2026-09-02`
Release-style frozen branch ref: `release/v0.2.0-submission-rc1`
Public base commit before freeze: `9d6c2904ba2167cdf70645332a1e93e3e2d0d4fd`

## Authoritative submission archive

The exact submission-primary reproducibility snapshot supplied with the manuscript as Supplementary Material is:

- file: `SWEVO_P8_REPRODUCIBILITY_SUPPLEMENT_2026-09-02.zip`
- SHA-256: `91914777f7d119a3abfbd345603a5398116f2884d87d7b32d054b36307fbc5d4`

Portal upload package:

- file: `SWEVO_P8_PORTAL_UPLOAD_PACKAGE_2026-09-02.zip`
- SHA-256: `b663cd74f1e3219d2de9dac17b47764b601f5a347e2304d4ece079db6fb8a5e1`

The authoritative evidence scope is 10,800 completed real-benchmark runs: 36 instances × 3 scenarios × 5 primary methods × 20 paired seeds. Historical A1/A2/A3 ablation rows are excluded from submission-primary inference.

The public-benchmark evidence is the overlapping full-horizon `[0,H]` regime special case, not a validation of sequential non-overlapping driver shifts. `compute_wh` is a model-based runtime proxy, not hardware-metered energy.

Historical `generated/`, `generated_real/`, `generated_pilot/`, and `generated_smoke/` namespaces were removed from the frozen submission/release refs; repository `main` was not modified.

## GitHub connector limitation

The connected GitHub write interface available during this freeze supports branches and file commits but does not expose creation of Git tag/release objects. Therefore the immutable archival identity for the submitted reproducibility package is the SHA-256 checksum above. `release/v0.2.0-submission-rc1` is a release-style branch ref and must not be described as a Git tag.
