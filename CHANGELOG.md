# C15 changes

- Preserved C14 ZIPs, checksums, 10,800 primary outcomes and all visible
  confirmatory preregistration/raw files; start/end SHA-256 comparison supplied.
- Reconciled the complete primary and ablation grids against raw run IDs.
- Recomputed original 108-cell inference and 10,795-row matched runtime
  sensitivity without rerunning the primary optimizer campaign.
- Added 36-instance Friedman/Wilcoxon-Holm sensitivity, signed rank-biserial
  effects and a 20,000-resample instance-cluster bootstrap (seed 20261116).
- Separated archive-free, portal-free core reproduction from portal checks;
  declared and pinned Jinja2 and other reproduction dependencies.
- Added atomic resumable operational records, source/manifest verification,
  host exclusion and interruption/duplicate/runtime tests without modifying
  solver definitions, budgets or seeds.
- Revised the canonical manuscript, reduced repeated narrative and moved
  detailed model/pseudocode and secondary results to appendices. Updated
  abstract, generated best-other comparison, highlights and graphical abstract.
- Restricted claims to feasible model-score quality against frozen internal
  comparators. No new field data, external solver, component conclusion or
  general convergence claim was added.
- Issued separately named C15 packages, internal/outer checksum ledgers and
  independent clean extraction, installation and reproduction records.
