# C15.1 build and evidence contract

Core inputs and the exact files rewritten by `--core` are explicitly listed
in `scripts/c151_core_contract.py`. A source CSV or frozen reference is never
classified as a generated result. The core records all templates, the frozen
C14 comparison reference, the dependency lock, configurations and generator
code hashes. It verifies unchanged frozen inputs after generation.

The Python lock is `requirements-c151.lock.txt`, including Jinja2 needed by
pandas LaTeX rendering. `pyproject.toml` requires Python >=3.11 and declares
compatible package minima. CPython 3.14.3 on Windows is the tested runtime.
Version declarations are not evidence that other runtimes were tested.

The separate manuscript source ZIP contains the article sources, BibTeX/BST
files and generated TeX inputs at the relative paths consumed by the article.
Compile from its `manuscript/` directory in the recorded TeX environment:

```text
pdflatex -interaction=nonstopmode -halt-on-error elsarticle-template-num.tex
bibtex elsarticle-template-num
pdflatex -interaction=nonstopmode -halt-on-error elsarticle-template-num.tex
pdflatex -interaction=nonstopmode -halt-on-error elsarticle-template-num.tex
pdflatex -interaction=nonstopmode -halt-on-error graphical_abstract_source.tex
```

The portal package carries `scripts/finalize_c15.py`, its contract module,
manuscript assets and the generated TeX inputs required by build provenance.
Install the same declared Python dependencies and run:

```text
python scripts/finalize_c15.py --portal
```

The reviewer supplement does not contain portal assets, so this is deliberately
a separate workflow. The public code branch omits manuscripts, raw primary
and secondary ledgers, private links and machine logs. Public unit tests use
synthetic fixtures; full `--core` reproduction requires the reviewer supplement.

For ablation auditing, file parse errors, invalid result rows, duplicate IDs,
unexpected IDs, missing IDs and interrupted checkpoints are reported separately.
Runtime compliance is separate from valid completion. Completed valid infeasible
solutions are retained with their flags and constraints. A run exceeding cap+10
is not called protocol-compliant because the process returned successfully.
Every four-method matched cell containing a timing excursion is excluded
together in the preregistered sensitivity. A final inferential call additionally
requires the source-locked original execution protocol.
