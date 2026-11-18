# C17.0 research code - partial campaign

C16.0 remains the authoritative research reference. This branch adds tested C17 event telemetry, independent route verification, dependence-aware analysis, a boundary-only selection variant and an original PyVRP 0.12.2 HGS adapter for a declared narrow homogeneous distance VRPTW.

The required confirmatory experiments remain incomplete. Development smoke checks do not establish algorithm performance. No manuscript, raw outcomes, benchmark copies or private operational records are included here. See [C17 code scope](docs/C17_PUBLIC.md).

Install requirements.lock.txt and run `python -m pytest -q -p no:cacheprovider`. Full evidence reanalysis and optimizer execution require the separate private author/reviewer package. The frozen ablation always loads its own unchanged source context, never this instrumented solver. No journal submission or acceptance is asserted.
