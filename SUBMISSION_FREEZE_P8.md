# Final submission freeze record

Manuscript: **Feasibility-First Enhanced Differential Evolution for Sustainability-Aware VRPTW with Shift-Indexed Operating Regimes**

Final owner-confirmation date: 2026-09-03
Submission branch: `submission-2026-09-02`
Release-style frozen branch ref: `release/v0.2.0-submission-rc1`
Public base commit before freeze: `9d6c2904ba2167cdf70645332a1e93e3e2d0d4fd`

## Authoritative submission archives

Final reproducibility supplement:

- file: `SWEVO_FINAL_REPRODUCIBILITY_SUPPLEMENT_2026-09-03.zip`
- SHA-256: `91d495ce7a64bda5b8fd98de0151818a86763e4b5caafaa27f38514d73129ae9`

Final portal upload package:

- file: `SWEVO_FINAL_PORTAL_UPLOAD_PACKAGE_2026-09-03.zip`
- SHA-256: `d964259975ef8b098a5f8dfcbaafc0845cc8735ec918f8921f40f8aece181a73`

The authoritative evidence scope is 10,800 completed real-benchmark runs: 36 instances × 3 scenarios × 5 primary methods × 20 paired seeds. Historical A1/A2/A3 ablation rows are excluded from submission-primary inference.

The public-benchmark evidence is the overlapping full-horizon `[0,H]` regime special case, not a validation of sequential non-overlapping driver shifts. `compute_wh` is a model-based runtime proxy, not hardware-metered energy.

Historical `generated/`, `generated_real/`, `generated_pilot/`, and `generated_smoke/` namespaces were removed from the frozen submission/release refs; repository `main` was not modified.

## Owner declarations confirmed

On 2026-09-03 the author explicitly confirmed: no specific grant funding; no competing interests; the listed single-author CRediT roles; no acknowledgements; and that the manuscript is not under consideration elsewhere. These declarations are incorporated into the final manuscript and portal package. No experimental outcome or inferential result was changed.

## GitHub connector limitation

The connected GitHub write interface supports branches and file commits but does not expose creation of Git tag/release objects. Therefore the immutable archival identity for the submitted reproducibility package is the SHA-256 checksum above. `release/v0.2.0-submission-rc1` is a release-style branch ref and must not be described as a Git tag.
