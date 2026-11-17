import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
from run_c15_ablation import execute,records,recover,classify,atomic
from analyze_c15_primary import cluster_ci,ids,METHODS

def test_normal_and_duplicate(tmp_path):
    p=SimpleNamespace(run_id='one',walltime_cap_s=1800)
    r=execute(p,lambda p:dict(run_id=p.run_id,runtime_sec=1),tmp_path,{})
    assert r['result_status']=='completed' and r['runtime_status']=='within_nominal'
    assert r['start_utc'] and r['end_utc'] and r['process_cpu_s']>=0
    with pytest.raises(ValueError):execute(p,lambda p:{},tmp_path,{})
def test_success_is_not_cap_compliance(tmp_path):
    p=SimpleNamespace(run_id='slow',walltime_cap_s=1800)
    r=execute(p,lambda p:dict(run_id=p.run_id,runtime_sec=20000),tmp_path,{})
    assert r['result_status']=='completed' and r['runtime_status']=='over_tolerance'
    assert classify(1805,1800)=='above_nominal_within_tolerance'
def test_interruption_and_crash_recovery(tmp_path):
    p=SimpleNamespace(run_id='stop',walltime_cap_s=1800)
    def stop(p):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):execute(p,stop,tmp_path,{})
    assert records(tmp_path)['stop']['result_status']=='interrupted'
    atomic(tmp_path/'crash.json',dict(run_id='crash',result_status='started'))
    recover(tmp_path);assert records(tmp_path)['crash']['result_status']=='interrupted'
def test_duplicate_records_fail(tmp_path):
    for name in ['a','b']:atomic(tmp_path/(name+'.json'),dict(run_id='same',result_status='completed'))
    with pytest.raises(ValueError):records(tmp_path)
def test_cluster_resamples_keep_scenarios_together():
    idx=pd.MultiIndex.from_product([['a','b','c'],['s1','s2','s3']],names=['instance_id','scenario_id'])
    table=pd.DataFrame({m:np.arange(9)+i for i,m in enumerate(METHODS)},index=idx)
    r=cluster_ci(table,reps=100,seed=1)
    assert np.allclose(r.ci_low,[-1,-2,-3,-4]) and np.allclose(r.ci_high,r.ci_low)
    pd.testing.assert_frame_equal(r,cluster_ci(table,reps=100,seed=1))
def test_actual_primary_audit():
    root=SCRIPTS.parent
    a=json.loads((root/'generated_submission/c15_primary_audit.json').read_text())
    assert a['primary_rows']==10800 and a['sensitivity_rows']==10795 and a['c14_inference_reconciled']
    t=pd.read_csv(root/'generated_submission/c15_instance_medians.csv');assert len(t)==36
