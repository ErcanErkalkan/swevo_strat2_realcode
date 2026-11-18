# C17 research code - incomplete scientific campaign

C16.0 remains the reference. This branch adds separate C17 telemetry and independent route auditing, post-hoc historical analysis code, a boundary-selection-only variant, and a pinned original PyVRP 0.12.2 HGS adapter for the explicit narrow homogeneous distance VRPTW. It contains no manuscript, raw experiment outcomes, private status or protected baseline snapshot.

The legacy BaseComparator interface exists to preserve historical formats; new C17 measurements must use scripts/run_c17_cohort.py. It does not replace absent telemetry with a final score/time. External C++ events not exposed by public hooks remain NA. New C17 code is never loaded into the frozen ablation process. A3 also removes route-ALNS endgame, so A3 alone cannot isolate boundary selection.

Tests: install requirements.lock.txt and run python -m pytest -q -p no:cacheprovider. Actual-ledger tests require the separate private reviewer supplement. Original HGS: pip install --target ../deps_hgs -r requirements-c17-external.lock.txt; source https://github.com/PyVRP/PyVRP/tree/v0.12.2, MIT, commit ea0c4211819edac6fd920413ad7508cc9ad56e0e. The current 0.14.1 solver uses ILS and was not mislabelled HGS.

No completed component-effect, full-budget external, binding-shift effectiveness, field saving, Pareto-superiority, unseen-test or above-400-customer claim is established by this code release. Software smoke checks are development-only. The complete private package contains manifests and executable continuation commands; this public code alone cannot reproduce private outcomes.
