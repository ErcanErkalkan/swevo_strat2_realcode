#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
import pandas as pd
from swevo_suite.paths import GENERATED, ensure_generated_dirs
from swevo_suite.stats import load_runs, pairwise_wilcoxon, friedman_by_instance

PRIMARY = 'j_scaled_final'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('src', nargs='?', default=str(GENERATED / 'master_runs.csv'))
    parser.add_argument('dst', nargs='?', default=str(GENERATED / 'master_stats.csv'))
    parser.add_argument('--control', default='EDE')
    parser.add_argument('--metric', default=PRIMARY)
    args = parser.parse_args()

    ensure_generated_dirs()
    df = load_runs(Path(args.src))
    pairwise = pairwise_wilcoxon(df, metric=args.metric, control_method=args.control)
    omnibus = friedman_by_instance(df, metric=args.metric)
    base = Path(args.dst)
    pairwise_path = base.with_name('pairwise_wilcoxon_effects.csv')
    omnibus_path = base.with_name('friedman_summary.csv')
    pairwise.to_csv(pairwise_path, index=False)
    omnibus.to_csv(omnibus_path, index=False)
    master = pd.concat([pairwise.assign(section='pairwise'), omnibus.assign(section='omnibus')], ignore_index=True, sort=False)
    master.to_csv(base, index=False)
    print(f"Wrote {base}")


if __name__ == '__main__':
    main()
