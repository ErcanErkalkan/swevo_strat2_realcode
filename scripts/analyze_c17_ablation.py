"""Nine-instance supplement; refuses incomplete frozen campaigns."""
from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from audit_ablation import audit
ROOT=Path(__file__).resolve().parents[1]


def exact_signed_rank(d):
    d=np.asarray(d,float);d=d[d!=0]
    if not len(d):return 1.
    r=rankdata(abs(d));observed=abs(np.sum(np.sign(d)*r))
    vals=[abs(np.sum(np.asarray(s)*r)) for s in itertools.product([-1,1],repeat=len(d))]
    return float(np.mean(np.asarray(vals)>=observed-1e-12))


def main():
    out=ROOT/'results/ablation';campaign=ROOT.parent/'frozen/EDE_SUSTAINABLE_VRPTW_C16.0/ablation'
    st=audit(campaign,out,ROOT/'results/ablation_execution')
    if not st['scientific_closure_ready']:raise SystemExit('INCOMPLETE: no component-effect inference; '+str(st['missing'])+' missing')
    data=pd.read_csv(out/'validated_records.csv');rows=[]
    caps={'small':1800,'medium':3600,'large':7200};keys=['instance_id','scenario_id','seed']
    bad=data[data.runtime_sec>data.tier.map(caps)+10][keys].drop_duplicates()
    clean=data.merge(bad.assign(exclude=True),on=keys,how='left');clean=clean[clean.exclude.isna()]
    for scope,d in [('full',data),('matched_four_method_sensitivity',clean)]:
        cell=d.groupby(['instance_id','scenario_id','method_id']).j_scaled_final.median().unstack()
        ids=sorted(d.instance_id.unique());ix=np.random.default_rng(20261117).integers(0,9,(20000,9));family=[]
        for m in ['A1_NoSeed','A2_NoJDE','A3_NoLNS']:
            contrasts=(cell.EDE-cell[m]).unstack('scenario_id').loc[ids].to_numpy();inst=np.median(contrasts,axis=1)
            b=np.median(contrasts[ix].reshape(20000,-1),axis=1)
            family.append({'scope':scope,'variant':m,'n':9,'median_instance_difference':float(np.median(inst)),'wins':int((inst<0).sum()),'ties':int((inst==0).sum()),'losses':int((inst>0).sum()),'p_exact':exact_signed_rank(inst),'cell_median_difference':float(np.median(contrasts)),'cell_cluster_low':float(np.quantile(b,.025)),'cell_cluster_high':float(np.quantile(b,.975))})
        for r,p in zip(family,multipletests([r['p_exact'] for r in family],method='holm')[1]):r['p_holm']=float(p)
        rows.extend(family)
    pd.DataFrame(rows).to_csv(out/'nine_instance_inference.csv',index=False)
if __name__=='__main__':main()
