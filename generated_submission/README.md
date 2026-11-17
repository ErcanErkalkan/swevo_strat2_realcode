# C15 evidence and its inputs

This directory contains C15 derived evidence, not the seed-level primary ledger.
The supplement provenance snapshot is preserved as
`c15_supplement_core_provenance.json`. It refers to original relative paths
and bytes in the supplement. Running the core command from this GitHub
checkout creates a separate local `c15_core_provenance.json`.

For reproduction from this checkout, copy these inputs from the exact C15
supplement and verify each SHA-256 value before running the core command.

| Input | SHA-256 |
| --- | --- |
| `generated_submission/master_runs.csv` | `b373820b3de0ce95d7daed4220502a19d0194c85eeba573a54a0ab9cfc75b86c` |
| `configs/experiment_manifest_submission_primary.csv` | `596feb0639ac1413d65f1c0a107edd979a9772d4c1044f79a1c7b0d441ebc687` |
| `configs/anytime_profile_locked.csv` | `b192dabe145d44f8a41151572bc55ee3197083c77c9d8243c6d83064e3aa02eb` |
| `configs/operational_shift_stress_locked.csv` | `aba44122f5e02d48ad44ee4ca4493ac90f781d4fcc921adad83d4f2ac04593d4` |
| `configs/parameter_sensitivity_locked.csv` | `21de27c26d4e030937f51b37d967759e54f981960cc7be6bfaaab4aada0fbf26` |
