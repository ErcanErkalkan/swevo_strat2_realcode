#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == 'scripts' else Path('/mnt/data/p1_gate_repo')
sys.path.insert(0, str(ROOT / 'src'))

from swevo_suite.stats import pairwise_wilcoxon, friedman_by_instance

PRIMARY_METHODS = ['EDE','StdDE','ALNS_MS','HGS_MS','ILS_MS']
J = 'j_scaled_final'
NATIVE_METRICS = ['cost_final','energy_final','co2_final']


def _reduction(a: float, b: float) -> float:
    return 100.0 * (b - a) / b if b != 0 else float('nan')


def _best_other(values: pd.Series) -> tuple[str, float]:
    candidates = values.drop(labels=['EDE'], errors='ignore')
    method = str(candidates.idxmin())
    return method, float(candidates.loc[method])


def build_scenario(df: pd.DataFrame) -> pd.DataFrame:
    block = (df.groupby(['instance_id','scenario_id','method_id'], as_index=False)[J].median())
    piv = block.groupby(['scenario_id','method_id'])[J].median().unstack('method_id')
    rows=[]
    for scenario, vals in piv.iterrows():
        best_m,best_v=_best_other(vals)
        ede=float(vals['EDE']); std=float(vals['StdDE'])
        rows.append({
            'scenario_id':scenario,
            'blocks':int(block[block['scenario_id']==scenario][['instance_id','scenario_id']].drop_duplicates().shape[0]),
            'ede_median_j':ede,
            'stdde_median_j':std,
            'best_other_method':best_m,
            'best_other_median_j':best_v,
            'ede_reduction_vs_stdde_pct':_reduction(ede,std),
            'ede_reduction_vs_best_other_pct':_reduction(ede,best_v),
        })
    return pd.DataFrame(rows)


def build_structure(df: pd.DataFrame) -> pd.DataFrame:
    block = (df.groupby(['instance_id','scenario_id','structure_class','method_id'], as_index=False)[J].median())
    piv = block.groupby(['structure_class','method_id'])[J].median().unstack('method_id')
    rows=[]
    for structure, vals in piv.iterrows():
        best_m,best_v=_best_other(vals)
        ede=float(vals['EDE']); std=float(vals['StdDE'])
        rows.append({
            'structure_class':structure,
            'blocks':int(block[block['structure_class']==structure][['instance_id','scenario_id']].drop_duplicates().shape[0]),
            'ede_median_j':ede,
            'stdde_median_j':std,
            'best_other_method':best_m,
            'best_other_median_j':best_v,
            'ede_reduction_vs_stdde_pct':_reduction(ede,std),
            'ede_reduction_vs_best_other_pct':_reduction(ede,best_v),
        })
    order={'clustered':0,'mixed':1,'random':2}
    out=pd.DataFrame(rows)
    return out.sort_values('structure_class', key=lambda x:x.map(order)).reset_index(drop=True)


def build_tier_native(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for tier in ['large','medium','small']:
        sub=df[df['tier']==tier]
        for metric in [J]+NATIVE_METRICS:
            vals=sub.groupby('method_id')[metric].median()
            best_m,best_v=_best_other(vals)
            ede=float(vals['EDE']); std=float(vals['StdDE'])
            rows.append({
                'tier':tier,'metric':metric,'ede_median':ede,'stdde_median':std,
                'reduction_vs_stdde_pct':_reduction(ede,std),
                'best_other_method':best_m,'best_other_median':best_v,
                'reduction_vs_best_other_pct':_reduction(ede,best_v),
            })
    return pd.DataFrame(rows)


def build_secondary_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for metric in NATIVE_METRICS:
        om=friedman_by_instance(df, metric=metric, accepted_only=False).copy()
        om.insert(0,'section','omnibus')
        pw=pairwise_wilcoxon(df, metric=metric, control_method='EDE', accepted_only=False).copy()
        pw.insert(0,'section','pairwise')
        rows.extend(om.to_dict('records'))
        rows.extend(pw.to_dict('records'))
    cols=['section','metric','blocks','k_methods','friedman_chi2','p_value','kendalls_w',
          'control_method','compare_method','n_pairs','median_diff_control_minus_compare',
          'wilcoxon_stat','rank_biserial','p_holm','summary_scope']
    return pd.DataFrame(rows).reindex(columns=cols)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('master_runs')
    ap.add_argument('--output-dir', required=True)
    args=ap.parse_args()
    df=pd.read_csv(args.master_runs)
    methods=sorted(df['method_id'].dropna().unique().tolist())
    if methods != sorted(PRIMARY_METHODS):
        raise SystemExit(f'Expected primary methods {sorted(PRIMARY_METHODS)}, got {methods}')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    build_scenario(df).to_csv(out/'robustness_by_scenario.csv',index=False)
    build_structure(df).to_csv(out/'robustness_by_structure.csv',index=False)
    build_tier_native(df).to_csv(out/'tier_native_endpoint_reductions.csv',index=False)
    build_secondary_stats(df).to_csv(out/'secondary_endpoint_stats.csv',index=False)
    print(f'Wrote robustness evidence to {out}')

if __name__=='__main__': main()
