"""Development only; no performance inference from these short optimizer checks."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swevo_suite.manifest import load_manifest
from dataclasses import replace
from run_c17_cohort import run
from run_ablation import atomic


def main():
    base=next(p for p in load_manifest(ROOT/'configs/experiment_manifest_submission_primary.csv') if p.instance_id=='C101')
    out=ROOT/'results/c17/development_smoke';out.mkdir(exist_ok=True)
    # All smoke inputs are a development instance, disjoint from the new
    # nine-instance comparison manifest; historical visibility acknowledged.
    for method in ['EDE','StdDE','ALNS_MS','HGS_MS','ILS_MS']:
        plan=replace(base,method_id=method,seed=901,walltime_cap_s=.25,eval_budget=500,run_id='DEV_C101_'+method+'_901',phase='development_only')
        result=run(plan);result['label']='DEVELOPMENT_CONTRACT_CHECK';atomic(out/(method+'.json'),result)
        assert result['independent_validation']['valid'],(method,result['independent_validation'])
        assert result['telemetry']['initial_score_valid']
        print(method,'independent validation PASS',result['telemetry']['full_candidate_evaluations'],flush=True)
    from external_hgs import execute
    for method in ['EDE_NARROW','PYVRP_HGS']:
        plan=replace(base,method_id=method,seed=901,walltime_cap_s=.25,eval_budget=500,run_id='DEV_C101_'+method+'_901',phase='development_only')
        result=execute(plan,method);result['label']='DEVELOPMENT_CONTRACT_CHECK';atomic(out/(method+'.json'),result)
        assert result['independent_validation']['valid'],(method,result['independent_validation'])
        print(method,'narrow independent validation PASS',flush=True)
if __name__=='__main__':main()
