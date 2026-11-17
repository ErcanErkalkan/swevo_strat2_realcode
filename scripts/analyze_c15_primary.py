"""C15 primary checks, 36-instance sensitivity and cluster bootstrap.

Runs on the immutable source CSV, without historical or portal dependencies.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests

METHODS=['EDE','StdDE','ALNS_MS','HGS_MS','ILS_MS']
CAPS={'small':1800,'medium':3600,'large':7200}
SEED=20261116
REPS=20000
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ids(df):
    return df.instance_id+'__'+df.scenario_id+'__'+df.method_id+'__seed'+df.seed.astype(int).map(lambda x:f'{x:02d}')
def infer(table,scope):
    assert table[METHODS].notna().all().all()
    fr=friedmanchisquare(*(table[m] for m in METHODS))
    rows=[dict(scope=scope,test='Friedman',compare_method='',n=len(table),statistic=float(fr.statistic),p_value=float(fr.pvalue),p_holm=None,kendalls_w=float(fr.statistic/(len(table)*4)))]
    pairs=[]
    for m in METHODS[1:]:
        d=(table.EDE-table[m]).to_numpy()
        test=wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto')
        nz=d[d!=0]; ranks=rankdata(abs(nz)); rbc=float(np.sum(ranks*np.sign(nz))/np.sum(ranks)) if len(nz) else 0.
        pairs.append(dict(scope=scope,test='Wilcoxon',compare_method=m,n=len(table),statistic=float(test.statistic),p_value=float(test.pvalue),median_diff=float(np.median(d)),rank_biserial_ede_minus_compare=rbc,ede_better_pct=float(np.mean(d<0)*100)))
    for row,p in zip(pairs,multipletests([r['p_value'] for r in pairs],method='holm')[1]): row['p_holm']=float(p)
    return rows+pairs
def cluster_ci(cells,reps=REPS,seed=SEED):
    instances=sorted(cells.index.get_level_values('instance_id').unique())
    scenarios=sorted(cells.index.get_level_values('scenario_id').unique())
    # One draw selects an instance and carries ALL its three scenarios together.
    ix=np.random.default_rng(seed).integers(0,len(instances),(reps,len(instances)))
    result=[]
    for method in METHODS[1:]:
        diff=(cells.EDE-cells[method]).unstack('scenario_id').loc[instances,scenarios].to_numpy()
        draws=np.median(diff[ix].reshape(reps,-1),axis=1)
        result.append(dict(compare_method=method,clusters=len(instances),cells=len(cells),median_cell_difference=float(np.median(diff)),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),bootstrap_reps=reps,seed=seed,unit='instance; three scenarios travel together'))
    return pd.DataFrame(result)
def analyze(root):
    out=root/'generated_submission'; out.mkdir(exist_ok=True)
    p=root/'generated_submission/master_runs.csv'; mp=root/'configs/experiment_manifest_submission_primary.csv'
    d=pd.read_csv(p); m=pd.read_csv(mp); m['run_id']=ids(m)
    assert len(d)==10800 and d.run_id.nunique()==10800 and not m.run_id.duplicated().any()
    assert set(d.run_id)==set(m.run_id),'Manifest/grid mismatch'
    joined=d.merge(m,on='run_id',suffixes=('','_manifest'),validate='one_to_one')
    for col in ['instance_id','scenario_id','method_id','seed','tier','eval_budget','customer_count']:
        assert (joined[col]==joined[col+'_manifest']).all(),col
    assert np.isfinite(d.select_dtypes('number').to_numpy()).all(),'Nonfinite primary fields'
    assert d.accepted_final.eq(1).all() and d.strict_duty_final.eq(1).all()
    for col in ['v_cap_final','v_tw_final','v_shift_final','overtime_sum_final','overtime_ratio_sum_final']: assert d[col].eq(0).all(),col
    joined['nominal_overrun']=joined.runtime_sec>joined.walltime_cap_s
    joined['over_tolerance']=joined.runtime_sec>joined.walltime_cap_s+10
    over=joined[joined.over_tolerance]
    assert len(over)==5 and set(over.method_id)==set(METHODS)
    assert set(over.instance_id)=={'R1_4_5'} and set(over.scenario_id)=={'S1_balanced'} and set(over.seed)=={20}
    # Exclude the full matched cell, including any compliant method in that cell.
    keys=['instance_id','scenario_id','seed']
    bad=set(map(tuple,over[keys].to_numpy()))
    keep=np.array([tuple(x) not in bad for x in d[keys].to_numpy()])
    clean=d[keep]; assert len(clean)==10795
    cells=d.groupby(['instance_id','scenario_id','method_id']).j_scaled_final.median().unstack()[METHODS]
    inst=d.groupby(['instance_id','method_id']).j_scaled_final.median().unstack()[METHODS]
    clean_cells=clean.groupby(['instance_id','scenario_id','method_id']).j_scaled_final.median().unstack()[METHODS]
    assert cells.shape==(108,5) and inst.shape==(36,5)
    stats=pd.DataFrame(infer(cells,'108_instance_scenario_cells')+infer(inst,'36_instances')+infer(clean_cells,'10795_runtime_sensitivity'))
    cistats=cluster_ci(cells)
    cells.to_csv(out/'c15_cell_medians.csv'); inst.to_csv(out/'c15_instance_medians.csv')
    stats.to_csv(out/'c15_inference.csv',index=False); cistats.to_csv(out/'c15_cluster_bootstrap.csv',index=False)
    over[['run_id','runtime_sec','walltime_cap_s']].to_csv(out/'c15_primary_runtime_deviations.csv',index=False)
    summary=d.groupby(['tier','method_id']).agg(runs=('run_id','size'),accepted_rate=('accepted_final','mean'),strict_rate=('strict_duty_final','mean'),score_median=('j_scaled_final','median'),runtime_median=('runtime_sec','median'),cost_median=('cost_final','median'),energy_median=('energy_final','median'),co2_median=('co2_final','median')).reset_index()
    summary.to_csv(out/'c15_method_tier_summary.csv',index=False)
    runtime=joined.groupby(['tier','method_id']).agg(runs=('run_id','size'),above_nominal=('nominal_overrun','sum'),above_tolerance=('over_tolerance','sum')).reset_index()
    runtime.to_csv(out/'c15_runtime_by_method_tier.csv',index=False)
    # Reconcile independently regenerated 108-cell statistics with C14 artifacts.
    old=pd.read_csv(out/'protocol_sensitivity.csv')
    new=stats[stats.scope=='108_instance_scenario_cells']
    assert np.isclose(new.iloc[0].statistic,old[(old.scope=='all_10800')&(old.section=='omnibus')].iloc[0].statistic,rtol=1e-10)
    for method in METHODS[1:]:
        a=new[new.compare_method==method].iloc[0]; b=old[(old.scope=='all_10800')&(old.compare_method==method)].iloc[0]
        assert np.isclose(a.p_holm,b.p_holm,rtol=1e-8,atol=1e-100),method
    sensitivity=stats[stats.scope=='10795_runtime_sensitivity'].reset_index(drop=True)
    for col in ['statistic','p_value','p_holm']:
        assert np.allclose(new[col].to_numpy(float),sensitivity[col].to_numpy(float),rtol=1e-10,atol=1e-100,equal_nan=True),col
    report=dict(schema='c15-primary-audit-v1',source_sha256=sha(p),manifest_sha256=sha(mp),primary_rows=len(d),unique_run_ids=d.run_id.nunique(),missing=0,duplicates=0,unexpected=0,accepted=int(d.accepted_final.sum()),strict=int(d.strict_duty_final.sum()),historical_over_tolerance=len(over),sensitivity_rows=len(clean),instances=len(inst),cells=len(cells),bootstrap_reps=REPS,bootstrap_seed=SEED,c14_inference_reconciled=True,sensitivity_invariant=True)
    (out/'c15_primary_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    pair=stats[(stats.scope=='36_instances')&(stats.test=='Wilcoxon')]
    latex=['\\begin{tabular}{lrrr}','\\toprule','Comparator & Median difference & $p_{Holm}$ & Rank biserial \\\\','\\midrule']
    for row in pair.itertuples():
        latex.append(f'{row.compare_method.replace("_", "-")} & {row.median_diff:.5f} & {row.p_holm:.3g} & {row.rank_biserial_ede_minus_compare:.4f} \\\\')
    latex+=['\\bottomrule','\\end{tabular}']
    (out/'table_c15_instance_sensitivity.tex').write_text('\n'.join(latex)+'\n')
    latex=['\\begin{tabular}{lrrr}','\\toprule','Comparator & Cell median difference & Cluster CI low & high \\\\','\\midrule']
    for row in cistats.itertuples():latex.append(f'{row.compare_method.replace("_","-")} & {row.median_cell_difference:.5f} & {row.ci_low:.5f} & {row.ci_high:.5f} \\\\')
    (out/'table_c15_cluster_ci.tex').write_text('\n'.join(latex+['\\bottomrule','\\end{tabular}'])+'\n')
    latex=['\\begin{tabular}{lrrrr}','\\toprule','Tier & E-DE & Std. DE & Best other & Reduction vs best \\\\','\\midrule']
    reductions={}
    for tier in ['small','medium','large']:
        group=summary[summary.tier==tier].set_index('method_id').score_median
        best=group.drop('EDE').idxmin();red=(1-group.EDE/group[best])*100;std=(1-group.EDE/group.StdDE)*100
        reductions[tier]=dict(best_method=best,best_reduction_pct=red,stdde_reduction_pct=std)
        latex.append(f'{tier.capitalize()} & {group.EDE:.3f} & {group.StdDE:.3f} & {best.replace("_","-")}: {group[best]:.3f} & {red:.1f}\\% \\\\')
    (out/'table_c15_tier_comparison.tex').write_text('\n'.join(latex+['\\bottomrule','\\end{tabular}'])+'\n')
    fr36=stats[(stats.scope=='36_instances')&(stats.test=='Friedman')].iloc[0]
    macros=[f'\\newcommand{{\\CFifteenFriedman}}{{{fr36.statistic:.2f}}}',f'\\newcommand{{\\CFifteenKendall}}{{{fr36.kendalls_w:.3f}}}']
    for tier,value in reductions.items():macros.append(f'\\newcommand{{\\CFifteenBest{tier.capitalize()}}}{{{value["best_reduction_pct"]:.1f}\\%}}')
    (out/'c15_macros.tex').write_text('\n'.join(macros)+'\n')
    (out/'c15_tier_effects.json').write_text(json.dumps(reductions,indent=2)+'\n')
    print(json.dumps(report,indent=2)); print(pair.to_string(index=False)); return report
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args();analyze(a.root.resolve())
