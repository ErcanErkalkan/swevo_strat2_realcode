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


def write_stats_bundle(
    df: pd.DataFrame,
    *,
    metric: str,
    control_method: str,
    master_path: Path,
    pairwise_path: Path,
    omnibus_path: Path,
    accepted_only: bool,
) -> None:
    pairwise = pairwise_wilcoxon(
        df,
        metric=metric,
        control_method=control_method,
        accepted_only=accepted_only,
    )
    omnibus = friedman_by_instance(df, metric=metric, accepted_only=accepted_only)
    pairwise.to_csv(pairwise_path, index=False)
    omnibus.to_csv(omnibus_path, index=False)
    master = pd.concat(
        [pairwise.assign(section='pairwise'), omnibus.assign(section='omnibus')],
        ignore_index=True,
        sort=False,
    )
    master.to_csv(master_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('src', nargs='?', default=str(GENERATED / 'master_runs.csv'))
    parser.add_argument('dst', nargs='?', default=str(GENERATED / 'master_stats.csv'))
    parser.add_argument('--control', default='EDE')
    parser.add_argument('--metric', default=PRIMARY)
    args = parser.parse_args()

    ensure_generated_dirs()
    df = load_runs(Path(args.src))
    base = Path(args.dst)
    accepted_base = base.with_name(f"{base.stem}_accepted_only{base.suffix}")

    write_stats_bundle(
        df,
        metric=args.metric,
        control_method=args.control,
        master_path=base,
        pairwise_path=base.with_name('pairwise_wilcoxon_effects.csv'),
        omnibus_path=base.with_name('friedman_summary.csv'),
        accepted_only=False,
    )
    write_stats_bundle(
        df,
        metric=args.metric,
        control_method=args.control,
        master_path=accepted_base,
        pairwise_path=accepted_base.with_name('pairwise_wilcoxon_effects_accepted_only.csv'),
        omnibus_path=accepted_base.with_name('friedman_summary_accepted_only.csv'),
        accepted_only=True,
    )
    print(f"Wrote {base}")
    print(f"Wrote {accepted_base}")


if __name__ == '__main__':
    main()
