# EDE Sustainable VRPTW: public code

Code, tests, documentation and derived summaries for the integrated E-DE benchmark study. Python >=3.11; tested CPython 3.14.3 / Windows 11 build 26200. Install `requirements.lock.txt`, then run `python -m pytest -q -p no:cacheprovider`.

The public snapshot excludes manuscript assets, raw primary/secondary outcomes, the primary run manifest, benchmark bytes and private operational records. The complete reviewer supplement supplies all core inputs. `python scripts/finalize.py --core` reproduces 41 outputs from that supplement without manuscript or archive dependencies. The public checkout alone cannot reproduce outputs requiring the withheld raw inputs.

All five methods finish feasible with zero overtime. E-DE obtains lower composite model scores among feasible solutions; this is not measured field savings or a universal solver-family ranking. Common nominal search caps differ from recorded elapsed time. The confirmatory component-ablation campaign remains open (48 valid of 1,080); no isolated component claim is supported. Synthetic regression fixtures are not scientific outcomes. See docs/ for methods, scope and reproduction. Historical release branches remain unchanged. No journal submission is asserted.
