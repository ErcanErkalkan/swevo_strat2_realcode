# Interim Statistical Analysis

- Source snapshot: `generated_real\interim_snapshot_master_runs.csv`
- Rows: 4576 total; 8 methods; 29 complete instance-scenario blocks.
- Primary metric: `j_scaled_final`; lower is better.
- Accepted and strict-duty rates are 100.00% for every method in the current snapshot.

## Method Summary

| method_id | runs | median_j | rel_gap_vs_ede_pct | median_cost | median_energy | median_co2 | median_wh | median_imp_per_wh |
| --------- | ---- | -------- | ------------------ | ----------- | ------------- | ---------- | --------- | ----------------- |
| EDE       | 572  | 0.741012 | 0                  | 2637.63     | 2010.46       | 998.696    | 26.0008   | 0.00643245        |
| HGS_MS    | 572  | 0.77753  | 4.6967             | 2671.54     | 2193.56       | 1060.03    | 26.0008   | 0                 |
| A1_NoSeed | 572  | 0.827463 | 10.4477            | 2906.24     | 2439.16       | 1137.15    | 26.0008   | 0.00717653        |
| ALNS_MS   | 572  | 0.851079 | 12.9326            | 2877.18     | 2639.82       | 1152.73    | 26.0025   | 0                 |
| A2_NoJDE  | 572  | 0.854284 | 13.2593            | 2966.08     | 2516.02       | 1143.98    | 26.0008   | 0.00284077        |
| ILS_MS    | 572  | 0.863753 | 14.2102            | 2958.51     | 2679.77       | 1173.56    | 26.0007   | 0                 |
| A3_NoLNS  | 572  | 0.893897 | 17.1032            | 3095.97     | 2764.71       | 1195.91    | 26.0009   | 0.000930996       |
| StdDE     | 572  | 0.908245 | 18.4127            | 3291.71     | 2787.54       | 1243.63    | 26.0008   | 0.00373608        |

## Omnibus Test

- Friedman test over method medians per instance-scenario block: chi2=166.126437, p=1.645e-32, Kendall_W=0.818357.
- Interpretation: method choice has a statistically significant and large effect on the primary score in this interim sample.

## Mean Ranks

| method_id | mean_rank |
| --------- | --------- |
| EDE       | 1.31034   |
| HGS_MS    | 1.75862   |
| A1_NoSeed | 3.41379   |
| ALNS_MS   | 4.7931    |
| A2_NoJDE  | 4.82759   |
| ILS_MS    | 5.58621   |
| A3_NoLNS  | 7.03448   |
| StdDE     | 7.27586   |

## Pairwise Wilcoxon Tests Versus EDE

| compare_method | n_pairs | wins_EDE_lower | losses_EDE_higher | win_rate_pct | median_diff_EDE_minus_compare | rank_biserial | p_holm      |
| -------------- | ------- | -------------- | ----------------- | ------------ | ----------------------------- | ------------- | ----------- |
| HGS_MS         | 572     | 458            | 114               | 80.0699      | -0.0244162                    | 0.601399      | 5.21302e-59 |
| A1_NoSeed      | 572     | 551            | 21                | 96.3287      | -0.0941862                    | 0.926573      | 2.30163e-93 |
| A2_NoJDE       | 572     | 566            | 6                 | 98.951       | -0.108645                     | 0.979021      | 1.55628e-94 |
| ALNS_MS        | 572     | 567            | 5                 | 99.1259      | -0.119682                     | 0.982517      | 1.55628e-94 |
| ILS_MS         | 572     | 535            | 37                | 93.5315      | -0.129278                     | 0.870629      | 3.65624e-92 |
| A3_NoLNS       | 572     | 572            | 0                 | 100          | -0.14861                      | 1             | 1.55628e-94 |
| StdDE          | 572     | 570            | 2                 | 99.6503      | -0.18912                      | 0.993007      | 1.55628e-94 |

## Scenario Medians

| scenario_id        | A1_NoSeed | A2_NoJDE | A3_NoLNS | ALNS_MS  | EDE      | HGS_MS   | ILS_MS   | StdDE    | best_method |
| ------------------ | --------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ----------- |
| S1_balanced        | 0.830699  | 0.859353 | 0.899249 | 0.854761 | 0.740553 | 0.786396 | 0.86754  | 0.912298 | EDE         |
| S2_peak_dirty      | 0.834016  | 0.856272 | 0.896479 | 0.852708 | 0.740418 | 0.779053 | 0.865881 | 0.910201 | EDE         |
| S3_mixed_fleet_arc | 0.817783  | 0.842086 | 0.888765 | 0.83784  | 0.741012 | 0.762555 | 0.85782  | 0.897589 | EDE         |

## Instance Best Methods

| instance_id | best_method |
| ----------- | ----------- |
| C101        | HGS_MS      |
| C104        | EDE         |
| C109        | EDE         |
| C201        | HGS_MS      |
| R101        | HGS_MS      |
| R104        | EDE         |
| R109        | EDE         |
| RC101       | EDE         |
| RC104       | EDE         |
| RC108       | EDE         |
