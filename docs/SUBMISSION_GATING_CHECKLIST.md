# Submission Gating Checklist

Do not submit before every item below is true.

1. Abstract is filled.
2. Highlights are filled.
3. Comparator list in manuscript == comparator list in manifest.
4. Every comparator has real run outputs in `generated/master_runs.csv`.
5. Accepted rows have zero final violations.
6. Strict-duty rows have zero overtime.
7. Search diagnostics are reported in separate tables.
8. Summary tables are auto-generated from `master_runs.csv`.
9. Stats tables are auto-generated from `master_stats.csv`.
10. Claim macros are generated from the outputs, not typed manually.
11. Every major claim appears in the claim-evidence map.
12. README reproduces the full run pipeline.
13. Tuning and test splits are fixed and documented.
14. Failed runs are reported, not hidden.
15. Conditional-on-accepted and unconditional summaries are both available.
