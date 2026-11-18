"""Refuse incomplete new paired grids; retain failures in audit denominators."""
from pathlib import Path
import json,sys
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]


def main():
    output=[]
    for label,folders,count in [('boundary_selection',['boundary_selection'],540),('external_narrow',['external_narrow'],180)]:
        rows=[]
        for folder in folders:
            for p in (ROOT/'results'/folder).glob('*.json'):
                r=json.loads(p.read_text());result=r.get('result',{});plan=result.get('plan',{})
                rows.append({'run_id':r.get('run_id'),'status':r.get('status'),'instance_id':plan.get('instance_id'),'scenario_id':plan.get('scenario_id'),'seed':plan.get('seed'),'method':result.get('selection',result.get('method',folder)),'score':result.get('final',{}).get('score'),'feasible':result.get('independent_validation',{}).get('valid',False),'source':str(p)})
        if len(rows)!=count or any(r['status']!='completed' for r in rows):raise SystemExit(f'{label}: incomplete/failed, {len(rows)}/{count} records; no effect inference')
        df=pd.DataFrame(rows);df.to_csv(ROOT/f'results/c17/{label}_full_audit.csv',index=False)
        methods=['boundary','random'] if label=='boundary_selection' else ['EDE_NARROW','PYVRP_HGS']
        keys=['instance_id','scenario_id','seed'];complete=df.pivot(index=keys,columns='method',values='feasible')
        if not set(methods).issubset(complete.columns) or not complete[methods].notna().all().all():raise ValueError('Unpaired identities')
        if not complete[methods].all().all():raise SystemExit(f'{label}: infeasibility disclosed; no unqualified score-superiority inference')
        cell=df.groupby(['instance_id','scenario_id','method']).score.median().unstack()
        contrasts=(cell[methods[0]]-cell[methods[1]]).groupby('instance_id').median()
        p=wilcoxon(contrasts,alternative='two-sided').pvalue if contrasts.ne(0).any() else 1.
        ix=np.random.default_rng(20261118).integers(0,len(contrasts),(20000,len(contrasts)));draws=np.median(contrasts.to_numpy()[ix],axis=1)
        output.append({'label':label,'expected_records':count,'recorded':len(df),'failed':0,'infeasible':0,'n_instances':len(contrasts),'median_difference':contrasts.median(),'ci_low':np.quantile(draws,.025),'ci_high':np.quantile(draws,.975),'p_single_comparison_family':p,'wins':int(contrasts.lt(0).sum()),'ties':int(contrasts.eq(0).sum()),'losses':int(contrasts.gt(0).sum())})
    pd.DataFrame(output).to_csv(ROOT/'results/c17/new_cohort_inference.csv',index=False)
if __name__=='__main__':main()
