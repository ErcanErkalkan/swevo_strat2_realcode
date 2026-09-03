# Final submission freeze record

Manuscript: **Feasibility-First Enhanced Differential Evolution for Sustainability-Aware VRPTW with Shift-Indexed Operating Regimes**

Final owner-confirmation date: 2026-09-03
Submission branch: `submission-2026-09-02`
Release-style frozen branch ref: `release/v0.2.0-submission-rc1`
Public base commit before freeze: `9d6c2904ba2167cdf70645332a1e93e3e2d0d4fd`

## Authoritative submission archives

Final reproducibility supplement:

- file: `SWEVO_FINAL_REPRODUCIBILITY_SUPPLEMENT_2026-09-03.zip`
- SHA-256: `3aad983271e0e1cb6e7b4c62e38949906861ccf156b6e74722b6d500b6576a95`

Final portal upload package:

- file: `SWEVO_FINAL_PORTAL_UPLOAD_PACKAGE_2026-09-03.zip`
- SHA-256: `0093cf1534ce65b863b2af20d3444bc5192654003e1b130b81876ad3ce471c27`

The authoritative evidence scope is 10,800 completed real-benchmark runs: 36 instances × 3 scenarios × 5 primary methods × 20 paired seeds. Historical A1/A2/A3 ablation rows are excluded from submission-primary inference.

The public-benchmark evidence is the overlapping full-horizon `[0,H]` regime special case, not a validation of sequential non-overlapping driver shifts. `compute_wh` is a model-based runtime proxy, not hardware-metered energy.

## Production hardware provenance

The production campaign was executed on a Dell Vostro 3888 desktop with an Intel Core i5-10400 CPU (6 physical cores / 12 threads, 2.90 GHz), 16 GB RAM (2×8 GB at 2400 MHz), integrated Intel UHD Graphics 630, Windows 11 Enterprise 64-bit build 26200, a 256 GB SK hynix NVMe SSD, and a 480 GB Toshiba TR200 SATA SSD. BIOS version was Dell 2.35.1 and the motherboard identifier was Dell 0RM5DR. The optimizer is CPU-based and did not use the integrated GPU for search computation.

The locked ledger does not embed a per-run hardware fingerprint. Runtime comparisons are therefore scoped to this documented workstation/protocol; no cross-platform runtime equivalence is claimed. Electrical power was not measured with hardware counters, so compute-energy remains a model-based proxy.

Historical `generated/`, `generated_real/`, `generated_pilot/`, and `generated_smoke/` namespaces were removed from the frozen submission/release refs; repository `main` was not modified.

## Owner declarations confirmed

On 2026-09-03 the author explicitly confirmed: no specific grant funding; no competing interests; the listed single-author CRediT roles; no acknowledgements; and that the manuscript is not under consideration elsewhere. These declarations are incorporated into the final manuscript and portal package. No experimental outcome or inferential result was changed.

## GitHub connector limitation

The connected GitHub write interface supports branches and file commits but does not expose creation of Git tag/release objects. Therefore the immutable archival identity for the submitted reproducibility package is the SHA-256 checksum above. `release/v0.2.0-submission-rc1` is a release-style branch ref and must not be described as a Git tag.
