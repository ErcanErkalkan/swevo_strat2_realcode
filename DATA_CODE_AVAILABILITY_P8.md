# Data and code availability — submission wording

The complete submission-primary reproducibility package is provided as Supplementary Material with the manuscript. It contains the source/configuration snapshot, submission-primary experiment manifest, the seed-level 10,800-run evidence ledger, analysis and validation scripts, generated evidence tables, claim-evidence mapping, and the reproducibility audit. The submitted archive is `SWEVO_P7_RELEASE_CANDIDATE_2026-09-02.zip`; its SHA-256 checksum is `7cb14f02da267f9dc3b322222551891759c5751d64f6eafb314d3759892a9d6d`.

A public GitHub submission-scope mirror is maintained at the `submission-2026-09-02` branch of `ErcanErkalkan/swevo_strat2_realcode`. The supplementary archive, rather than historical generated outputs in older repository history, is the authoritative evidence snapshot for the submitted manuscript. Historical A1/A2/A3 exploratory runs are excluded from submission-primary inference.

The locked ledger does not preserve a complete immutable CPU/RAM/OS record for the original production executions. Accordingly, the package supports analysis reproduction from the locked outcomes but does not claim cross-platform reproduction of wall-clock runtimes or hardware-metered energy consumption.
