# EDE Sustainable VRPTW — C16.0

This branch is the public code reference for C16.0. C16.0 consolidates the
verified manuscript, frozen reproducibility evidence and ablation records under
one release identifier. The solver, methods, budgets, seeds, and recorded
outcomes are unchanged. This checkout contains only public code, tests,
documentation and derived summaries.

Tested with CPython 3.12.14. Install the pinned environment and run the tests
from the repository root:

    python -m pip install -r requirements-c151.lock.txt
    python -m pytest -q -p no:cacheprovider

The complete author package includes the separate reviewer supplement and the
primary raw ledger. Use its extracted workspace for full evidence reproduction
with `python scripts/finalize_c15.py --core`. This public checkout intentionally
does not contain the manuscript, raw experimental records, benchmark copies, or
operational notes. The name of the existing dependency lock is preserved to
keep the reproducibility inputs unchanged.

The integrated E-DE result concerns lower composite model scores among feasible
solutions under overlapping full-horizon regimes. It establishes no field
energy savings, universal solver superiority, or isolated component effect.
The confirmatory campaign is open (48 of 1,080 records in the consolidated
snapshot). No journal submission or acceptance is asserted. Earlier branches
are retained for history and are not the current public reference.
