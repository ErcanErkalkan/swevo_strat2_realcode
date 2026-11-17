# C15.1

- Reproduced the two C15 ablation defects with failing regression tests, then
  fixed empty-campaign handling and invalid full-grid acceptance.
- Separated coverage, record validation, incomplete attempts, runtime compliance
  and matched-cell scientific closure. Retained valid infeasible solutions as
  scientific records. Preserved solver runtime, outer wall time and CPU fields.
- Added synthetic-only tests for malformed values, metadata, duplicate and
  unexpected IDs, JSON checkpoints and four-method runtime exclusions.
- Kept all C14/C15 release files and raw primary/ablation/preregistration inputs
  unchanged. No remaining long run or new pilot was launched.
- Regenerated 108-cell/36-instance inference, instance-cluster bootstrap and
  10,795-row runtime sensitivity from the unchanged primary CSV.
- Classified the frozen C14 comparison reference and templates as provenance
  inputs. Enumerated actual generated outputs instead of collecting a directory.
- Separated byte hashes, newline changes and numeric equivalence. Declared
  Python >=3.11 consistently with the pinned runtime dependencies.
- Corrected duplicated appendix-reference prefixes without changing labels;
  generated two-decimal best-other score reductions in abstract and conclusion.
- Curated a reviewer supplement and a separate public code tree with explicit
  allowlists, excluding machine logs, private links and unnecessary operational
  snapshots. Rebuilt and independently verified PDF/source/supplement/portal.
