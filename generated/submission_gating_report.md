# Submission Gating Report

- pass: 12
- fail: 0
- manual_review_needed: 3

| Item | Status | Notes |
| --- | --- | --- |
| 1. Abstract is filled | manual_review_needed | Manuscript source is not available in this repo. |
| 2. Highlights are filled | manual_review_needed | Manuscript source is not available in this repo. |
| 3. Comparator list in manuscript == comparator list in manifest | manual_review_needed | Manifest methods can be checked locally, but manuscript text must be verified manually. |
| 4. Every comparator has real run outputs in generated/master_runs.csv | pass | Methods present: A1_NoSeed, A2_NoJDE, A3_NoLNS, ALNS_MS, EDE, HGS_MS, ILS_MS, StdDE; rows=800 |
| 5. Accepted rows have zero final violations | pass | All accepted rows are clean. |
| 6. Strict-duty rows have zero overtime | pass | All strict-duty rows have zero overtime. |
| 7. Search diagnostics are reported in separate tables | pass | Separate LaTeX tables are present. |
| 8. Summary tables are auto-generated from master_runs.csv | pass | summary_by_method.csv is present. |
| 9. Stats tables are auto-generated from master_stats.csv | pass | master_stats.csv is present. |
| 10. Claim macros are generated from the outputs, not typed manually | pass | generated/claim_macros.tex is present. |
| 11. Every major claim appears in the claim-evidence map | pass | All template claim ids are covered by generated claim_evidence_map.csv. |
| 12. README reproduces the full run pipeline | pass | README contains the core pipeline commands. |
| 13. Tuning and test splits are fixed and documented | pass | docs/EXPERIMENT_SPLITS.md documents the fixed split roles. |
| 14. Failed runs are reported, not hidden | pass | Failure-report files are present in generated/. |
| 15. Conditional-on-accepted and unconditional summaries are both available | pass | Both unconditional and accepted-only summaries/stats are present. |
