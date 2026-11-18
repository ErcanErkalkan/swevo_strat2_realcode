"""Read-only historical audits and explicitly post-hoc instance-cluster inference."""
from pathlib import Path
import sys,json,hashlib,itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def historical_audit(root=ROOT):
    src=root/'data/primary/runs.csv';d=pd.read_csv(src);o=root/'results/c17';o.mkdir(parents=True,exist_ok=True)
    a=d[['run_id','method_id','j_scaled_init','j_scaled_final','time_to_first_feasible_sec','runtime_sec','n_repair_attempts','n_repair_success','notes']].copy()
    a['initial_note_present']=a.notes.str.contains(r'(?:^|\|)\s*init_best=',regex=True)
    a['initial_valid']=a.initial_note_present
    a['initial_reason']=np.where(a.initial_valid,'explicit_init_best_note; rounded historical value','no_initial_measurement_note; final-score fallback or unverified')
    a['derived_initial_score']=a.j_scaled_init.where(a.initial_valid)
    a['first_feasible_equals_runtime']=np.isclose(a.time_to_first_feasible_sec,a.runtime_sec,rtol=0,atol=1e-9)
    a['first_feasible_valid']=~a.first_feasible_equals_runtime & a.time_to_first_feasible_sec.ge(0) & a.time_to_first_feasible_sec.le(a.runtime_sec)
    # HGS/ILS historical rows have no event provenance. Do not infer measured
    # first-feasible values solely from a non-equality with the fallback.
    a.loc[a.method_id.isin(['HGS_MS','ILS_MS']),'first_feasible_valid']=False
    a['derived_first_feasible_s']=a.time_to_first_feasible_sec.where(a.first_feasible_valid)
    a['first_feasible_reason']=np.where(a.first_feasible_valid,'historical kernel event; not independently timed','runtime fallback or missing event provenance')
    a['repair_valid']=False
    a['repair_reason']='aggregate historical counters omit initialization-local SearchStats; zero is not a verified full-run zero'
    a['derived_repair_attempts']=pd.Series([pd.NA]*len(a),dtype='Int64')
    a['derived_repair_successes']=pd.Series([pd.NA]*len(a),dtype='Int64')
    a['imp_per_wh_valid']=False
    a['final_score_status']='retained separately; original feasibility flags not independent route audit'
    a['source_file']='data/primary/runs.csv';a['source_row']=np.arange(2,len(a)+2)
    a.to_csv(o/'historical_diagnostic_validity.csv',index=False,na_rep='NA')
    summary=a.groupby('method_id').agg(rows=('run_id','size'),initial_valid=('initial_valid','sum'),first_feasible_valid=('first_feasible_valid','sum'),fallback_time_equals_runtime=('first_feasible_equals_runtime','sum'),repair_valid=('repair_valid','sum'),zero_repair=('n_repair_attempts',lambda s:int(s.eq(0).sum())),nonzero_repair=('n_repair_attempts',lambda s:int(s.ne(0).sum()))).reset_index()
    summary.to_csv(o/'historical_diagnostic_summary.csv',index=False)
    protected=json.loads((root/'control/BASELINE_PRESERVATION.json').read_text())['protected']
    expected=next(v for k,v in protected.items() if k.replace('\\','/').endswith('/data/primary/runs.csv'))
    assert sha(src)==expected,'Historical raw input changed'
    return summary


def instance_analysis(root=ROOT):
    d=pd.read_csv(root/'data/primary/runs.csv');m=pd.read_csv(root/'configs/experiment_manifest_submission_primary.csv');o=root/'results/c17'
    caps={'small':1800,'medium':3600,'large':7200}
    bad=d[d.runtime_sec>d.tier.map(caps)+10][['instance_id','scenario_id','seed']].drop_duplicates()
    clean=d.merge(bad.assign(exclude=True),on=['instance_id','scenario_id','seed'],how='left');clean=clean[clean.exclude.isna()]
    rows=[];effects=[];lookup=[]
    for scope,df in [('all_10800',d),('matched_10795',clean)]:
        cell=df.groupby(['instance_id','scenario_id','method_id']).j_scaled_final.median().unstack('method_id')
        pooled=df.groupby(['instance_id','method_id']).j_scaled_final.median().unstack('method_id')
        ids=sorted(df.instance_id.unique());ix=np.random.default_rng(20261118).integers(0,len(ids),(20000,len(ids)))
        family=[]
        for method in ['StdDE','ALNS_MS','HGS_MS','ILS_MS']:
            diff=(cell.EDE-cell[method]).unstack('scenario_id').loc[ids].to_numpy();inst=np.median(diff,axis=1)
            test=wilcoxon(inst,zero_method='wilcox',alternative='two-sided',method='auto')
            nz=inst[abs(inst)>1e-12];ranks=rankdata(abs(nz));rbc=float(np.sum(ranks*np.sign(nz))/ranks.sum()) if len(nz) else 0.
            boot_inst=np.median(inst[ix],axis=1);boot_cell=np.median(diff[ix].reshape(20000,-1),axis=1)
            row={'scope':scope,'method':method,'n_instances':36,'n_cells':108,'median_instance_contrast':float(np.median(inst)),
                 'ci_instance_low':float(np.quantile(boot_inst,.025)),'ci_instance_high':float(np.quantile(boot_inst,.975)),
                 'median_cell_contrast':float(np.median(diff)),'ci_cell_low':float(np.quantile(boot_cell,.025)),'ci_cell_high':float(np.quantile(boot_cell,.975)),
                 'old_pooled_60_seed_median_contrast':float(np.median(pooled.EDE-pooled[method])),
                 'rank_biserial':rbc,'wins':int((inst< -1e-12).sum()),'ties':int((abs(inst)<=1e-12).sum()),'losses':int((inst>1e-12).sum()),'p':float(test.pvalue),
                 'bootstrap_reps':20000,'bootstrap_seed':20261118,'label':'POST_HOC_HISTORICAL'}
            family.append(row)
            for i,name in enumerate(ids):effects.append({'scope':scope,'method':method,'instance_id':name,'median_scenario_contrast':inst[i]})
        for row,p in zip(family,multipletests([r['p'] for r in family],method='holm')[1]):row['p_holm']=float(p)
        rows.extend(family)
        ids_to_cells={}
        for rec in df.itertuples():lookup.append({'table':'instance_effects','scope':scope,'instance_id':rec.instance_id,'scenario_id':rec.scenario_id,'method_id':rec.method_id,'run_id':rec.run_id,'source':'data/primary/runs.csv'})
    table=pd.DataFrame(rows);table.to_csv(o/'instance_inference.csv',index=False);pd.DataFrame(effects).to_csv(o/'instance_effects.csv',index=False);pd.DataFrame(lookup).to_csv(o/'table_run_lineage.csv',index=False)
    assert len(clean)==10795
    tex=['\\begin{tabular}{lrrrr}','\\toprule','Comparator & Median & 95\\% CI & W/T/L & Holm $p$ \\\\','\\midrule']
    for r in table[table.scope=='all_10800'].itertuples():tex.append(f'{r.method.replace("_","-")} & {r.median_instance_contrast:.5f} & [{r.ci_instance_low:.5f}, {r.ci_instance_high:.5f}] & {r.wins}/{r.ties}/{r.losses} & {r.p_holm:.3g} \\\\')
    (o/'table_instance_inference.tex').write_text('\n'.join(tex+['\\bottomrule','\\end{tabular}'])+'\n')
    provenance={'label':'POST_HOC_HISTORICAL','analysis_command':'python scripts/analyze_c17.py','script_sha256':sha(Path(__file__)),'input_sha256':sha(root/'data/primary/runs.csv'),'outputs':{p.name:sha(p) for p in o.glob('*.csv')},'estimand_note':'Median of three scenario-wise seed-median contrasts differs from a contrast of pooled 60-seed medians and from median over 108 cell contrasts. Bootstrap draws instances and carries all three scenarios.'}
    (o/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return table


if __name__=='__main__':
    print(historical_audit().to_string(index=False));print(instance_analysis().to_string(index=False))
