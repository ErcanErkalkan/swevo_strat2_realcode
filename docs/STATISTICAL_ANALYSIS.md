# Statistical addendum (post-campaign sensitivity)

The primary solver grid and original scientific preregistrations are unchanged.
This is a post-campaign sensitivity analysis, not a prospective preregistration
of already observed primary outcomes.

The original 108 instance-by-scenario medians are reproduced for continuity.
Their three scenarios repeat each of 36 underlying instances, so independent
problem sampling at the cell level is not asserted. Stochastic seeds remain
nested within each cell; individual seeds are not treated as independent
benchmark instances.

For each instance/method, pool all three scenarios and 20 seeds (60 scores)
and take the median. Apply Friedman on the 36 matched rows. Compare EDE
with StdDE, ALNS_MS, HGS_MS and ILS_MS using two-sided Wilcoxon signed-rank
tests (`zero_method=wilcox`, SciPy `method=auto`), with Holm across those four
comparisons. Signed rank-biserial effects use nonzero differences ranked by
absolute magnitude; negative EDE-minus-comparator favors EDE.

For uncertainty on the median of the 108 cell-level differences, draw 36
instance indices with replacement and carry all three scenarios of each
drawn instance together. Take the median over the resulting 108 differences.
Use 20,000 draws, fixed seed 20261116 and percentile 2.5/97.5% intervals.
The identical resampling indices are used for all comparators. This estimand
differs from the median of the 36 pooled-instance differences and is labelled
separately in the manuscript tables. No bootstrap CI is presented as a new
confirmatory hypothesis test.

Historical primary runtime sensitivity excludes all five methods in the
R1_4_5/S1_balanced/seed20 match and leaves the raw ledger untouched.
Confirmatory-ablation runtime sensitivity excludes the complete four-method
C104/S1_balanced/seed06 match, not just its two offending rows. No partial
ablation comparison is performed. No replacement rule or compensation run
has been executed; an eventual rule must be recorded before a complete
four-method matched rerun in a separate namespace.
