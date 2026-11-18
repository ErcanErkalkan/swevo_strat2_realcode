#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

EXPECTED_CONFIGS=['BASE','LNS5','LNS20','REPAIR8','REPAIR20']
EXPECTED_INSTANCES=['C101','R101','RC101']
EXPECTED_SEEDS=[1,2,3]
EXPECTED_SCENARIO='S3_mixed_fleet_arc'

def sha256(path:Path)->str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def holm_adjust(raw):
    raw=np.asarray(raw,float); order=np.argsort(raw); adj=np.empty_like(raw); running=0.0; m=len(raw)
    for rank,idx in enumerate(order):
        val=min(1.0, raw[idx]*(m-rank)); running=max(running,val); adj[idx]=running
    return adj

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ledger',default='data/secondary/parameter_sensitivity_locked.csv'); ap.add_argument('--out',default='results/primary'); a=ap.parse_args()
    ledger=Path(a.ledger); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(ledger)
    required={'instance_id','scenario_id','method_id','seed','accepted_final','strict_duty_final','v_cap_final','v_tw_final','v_shift_final','j_scaled_final','lns_period','repair_budget','runtime_sec'}
    missing=required-set(df.columns)
    if missing: raise SystemExit(f'missing columns: {sorted(missing)}')
    if len(df)!=45: raise SystemExit(f'expected 45 rows, found {len(df)}')
    if set(df.instance_id.astype(str))!=set(EXPECTED_INSTANCES): raise SystemExit('instance grid mismatch')
    if set(df.method_id.astype(str))!=set(EXPECTED_CONFIGS): raise SystemExit('config grid mismatch')
    if set(df.seed.astype(int))!=set(EXPECTED_SEEDS): raise SystemExit('seed grid mismatch')
    if set(df.scenario_id.astype(str))!={EXPECTED_SCENARIO}: raise SystemExit('scenario mismatch')
    if df[['instance_id','method_id','seed']].duplicated().any(): raise SystemExit('duplicate block/config/seed rows')
    if int(df.accepted_final.sum())!=45: raise SystemExit('not all runs accepted')
    if int(df.strict_duty_final.sum())!=45: raise SystemExit('not all runs strict duty')
    for c in ['v_cap_final','v_tw_final','v_shift_final']:
        if float(pd.to_numeric(df[c]).abs().max())>1e-9: raise SystemExit(f'nonzero violation in {c}')
    p=df.pivot(index=['instance_id','seed'],columns='method_id',values='j_scaled_final').sort_index()
    if p.shape!=(9,5) or p.isna().any().any(): raise SystemExit(f'incomplete paired grid {p.shape}')
    base=p['BASE']
    summary=[]
    for m in EXPECTED_CONFIGS:
        vals=p[m]; rel=100.0*(vals-base)/base if m!='BASE' else np.zeros(len(vals))
        summary.append({
            'config':m,'runs':int((df.method_id==m).sum()),'paired_blocks':len(vals),
            'hard_feasible_runs':int(df.loc[df.method_id==m,'accepted_final'].sum()),
            'strict_duty_runs':int(df.loc[df.method_id==m,'strict_duty_final'].sum()),
            'median_score':float(vals.median()),
            'median_pct_delta_vs_base':float(np.median(rel)),
            'max_abs_pct_delta_vs_base':float(np.max(np.abs(rel))),
            'wins_vs_base':int((vals<base-1e-12).sum()) if m!='BASE' else 0,
            'ties_vs_base':int((np.abs(vals-base)<=1e-12).sum()) if m!='BASE' else len(vals),
            'losses_vs_base':int((vals>base+1e-12).sum()) if m!='BASE' else 0,
            'lns_period':int(df.loc[df.method_id==m,'lns_period'].iloc[0]),
            'repair_budget':int(df.loc[df.method_id==m,'repair_budget'].iloc[0]),
        })
    sdf=pd.DataFrame(summary); sdf.to_csv(out/'parameter_sensitivity_summary.csv',index=False)
    stat,pv=friedmanchisquare(*[p[m].values for m in EXPECTED_CONFIGS])
    stats=[{'section':'omnibus','compare_config':'ALL','statistic':float(stat),'p_raw':float(pv),'p_holm':np.nan,'n_blocks':9}]
    raw=[]
    for m in EXPECTED_CONFIGS[1:]:
        try: w,pp=wilcoxon(p[m].values,base.values,zero_method='zsplit',alternative='two-sided')
        except ValueError: w,pp=0.0,1.0
        raw.append((m,float(w),float(pp)))
    adj=holm_adjust([x[2] for x in raw])
    for (m,w,pp),pa in zip(raw,adj): stats.append({'section':'pairwise_vs_BASE','compare_config':m,'statistic':w,'p_raw':pp,'p_holm':float(pa),'n_blocks':9})
    st=pd.DataFrame(stats); st.to_csv(out/'parameter_sensitivity_stats.csv',index=False)
    # latex table
    lines=[\
        r'\begin{table}[H]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
        r'\renewcommand{\arraystretch}{1.12}', r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Configuration & Feasible & Median $J$ & Median $\Delta$ vs base (\%) & Max $|\Delta|$ (\%) & W/T/L \\', r'\midrule'
    ]
    for _,r in sdf.iterrows():
        cfg=str(r['config']).replace('_','\\_')
        lines.append(f"{cfg} & {int(r['hard_feasible_runs'])}/{int(r['runs'])} & {r['median_score']:.3f} & {r['median_pct_delta_vs_base']:.3f} & {r['max_abs_pct_delta_vs_base']:.3f} & {int(r['wins_vs_base'])}/{int(r['ties_vs_base'])}/{int(r['losses_vs_base'])}" + r" \\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Secondary short-budget local parameter-robustness screen on C101, R101, and RC101 under S3 (three paired seeds; 3-s wall-clock cap). Lower $J$ is better. W/T/L denotes wins/ties/losses versus the default E-DE configuration. This screen is descriptive and does not replace full-budget sensitivity or tuning studies.}',r'\label{tab:parameter_sensitivity_secondary}',r'\end{table}']
    (out/'table_parameter_sensitivity.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    payload={'schema':'ede-secondary-parameter-sensitivity-v1','input':{'path':ledger.as_posix(),'bytes':ledger.stat().st_size,'sha256':sha256(ledger)},'design':{'instances':EXPECTED_INSTANCES,'scenario':EXPECTED_SCENARIO,'seeds':EXPECTED_SEEDS,'configs':EXPECTED_CONFIGS,'walltime_cap_s':3,'eval_budget':300,'scope':'secondary_local_short_budget_not_full_tuning'},'outputs':{}}
    for name in ['parameter_sensitivity_summary.csv','parameter_sensitivity_stats.csv','table_parameter_sensitivity.tex']:
        q=out/name; payload['outputs'][name]={'bytes':q.stat().st_size,'sha256':sha256(q)}
    (out/'parameter_sensitivity_evidence_provenance.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(f"parameter_sensitivity=PASS runs=45 feasible=45 friedman_p={pv:.6g} min_holm={np.nanmin(st.p_holm.values):.6g}")
if __name__=='__main__': main()
