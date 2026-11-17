# C15.1 public code and derived summaries

Python >=3.11 is required; CPython 3.14.3 on Windows was tested.
Install `requirements-c151.lock.txt`, then run `python -m pytest -q -p no:cacheprovider`.
Synthetic fixtures exercise empty/malformed ablation grids and record validity;
they are not scientific outcomes. The actual confirmatory campaign remains open.

The public tree contains code, tests, documentation and derived summaries only.
The manuscript and raw primary/secondary outcomes are distributed separately in
the C15.1 reviewer supplement. Run `scripts/finalize_c15.py --core` from that
complete supplement; this public checkout does not itself contain all raw inputs.
See scripts/c151_release_contract.py RAW_INPUTS for the exact supplement-only
inputs. The core requires neither the historical archive nor portal documents.

The integrated E-DE result concerns lower composite model scores among feasible
solutions under overlapping full-horizon regimes. It establishes no field energy
savings, universal solver superiority or isolated component effect. This revision
fixes scientific-record validation and input/output provenance without changing
the frozen solver, budgets, seeds or raw outcomes. Old release branches remain
unchanged. No journal submission or acceptance is asserted.
