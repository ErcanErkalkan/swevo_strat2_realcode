# P8 submission freeze record

Manuscript: **Feasibility-First Enhanced Differential Evolution for Sustainability-Aware VRPTW with Shift-Indexed Operating Regimes**

Freeze date: 2026-09-02
Submission branch: `submission-2026-09-02`
Public base commit before freeze: `9d6c2904ba2167cdf70645332a1e93e3e2d0d4fd`

## Authoritative submission archive

The exact submission-primary reproducibility snapshot is the P7 full release candidate supplied with the submission as supplementary material:

- file: `SWEVO_P7_RELEASE_CANDIDATE_2026-09-02.zip`
- SHA-256: `7cb14f02da267f9dc3b322222551891759c5751d64f6eafb314d3759892a9d6d`

Portal upload package:

- file: `SWEVO_P7_PORTAL_UPLOAD_PACKAGE_2026-09-02.zip`
- SHA-256: `dfe3631c79ad2aff89a4ec5e2aeeebce48cd26dfe1b94adae7d188c9ba6df324`

The authoritative evidence scope is 10,800 completed real-benchmark runs: 36 instances × 3 scenarios × 5 primary methods × 20 paired seeds. Historical A1/A2/A3 ablation rows are excluded from submission-primary inference.

The public-benchmark evidence is the overlapping full-horizon `[0,H]` regime special case, not a validation of sequential non-overlapping driver shifts. `compute_wh` is a model-based runtime proxy, not hardware-metered energy.

## GitHub connector limitation

The connected GitHub write interface available during this freeze supports branches and file commits but does not expose creation of Git tag/release objects. Therefore the immutable archival identity for the submitted reproducibility package is the SHA-256 checksum above. A release-style branch ref may point to the final frozen commit, but it must not be described as a Git tag.
