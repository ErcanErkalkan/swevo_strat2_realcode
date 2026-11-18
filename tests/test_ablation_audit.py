"""Synthetic validation fixtures only; never optimizer or scientific outcomes."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_ablation import audit
from analyze_primary import ids

@pytest.fixture
def campaign(tmp_path):
    root=tmp_path/'synthetic_campaign';root.mkdir();(root/'results').mkdir()
    plans=[]
    methods=['EDE','A1_NoSeed','A2_NoJDE','A3_NoLNS']
    for i in range(9):
        tier=['small','medium','large'][i//3]
        for scenario in range(3):
            for seed in range(1,11):
                for method in methods:
                    plans.append(dict(phase='confirmatory_ablation',benchmark_family='synthetic_test',customer_count=[100,200,400][i//3],instance_id=f'TEST{i}',structure_class=['clustered','random','mixed'][i%3],scenario_id=f'TEST_SCENARIO{scenario}',method_id=method,method_group='test',ablation_flag='test',seed=seed,tier=tier,eval_budget=[25050,40080,80100][i//3],walltime_cap_s=[1800,3600,7200][i//3],status='planned'))
    manifest=pd.DataFrame(plans);manifest.to_csv(root/'confirmatory_ablation_manifest_1080_cli.csv',index=False)
    results=manifest.drop(columns=['walltime_cap_s','status','method_group','ablation_flag']).copy()
    results['run_id']=ids(results)
    for name,value in dict(j_scaled_final=.75,runtime_sec=1.,accepted_final=1,strict_duty_final=1,v_cap_final=0.,v_tw_final=0.,v_shift_final=0.,overtime_sum_final=0.,overtime_ratio_sum_final=0.).items():results[name]=value
    return root,results,tmp_path/'audit'
def write(root,df):df.to_csv(root/'results/results_test.csv',index=False)

def test_empty_campaign(campaign):
    root,_,out=campaign
    status=audit(root,out)
    assert status['recorded_rows']==0 and status['missing']==1080
    assert status['complete'] is False
    assert len(pd.read_csv(out/'missing.csv'))==1080

def test_malformed_full_grid(campaign):
    root,df,out=campaign;df.loc[0,'runtime_sec']=-1;df.loc[1,'j_scaled_final']=np.nan;write(root,df)
    status=audit(root,out)
    assert status['complete'] is False
    assert status['invalid_records']==2 and status['coverage_complete'] is True

def test_valid_partial(campaign):
    root,df,out=campaign;write(root,df.iloc[:48]);s=audit(root,out)
    assert s['matched_unique']==48 and s['missing']==1032 and s['invalid_records']==0
    assert not s['complete'] and s['record_validation_passed']

def test_valid_infeasible_is_a_scientific_record(campaign):
    root,df,out=campaign;df.loc[0,['accepted_final','strict_duty_final','v_tw_final','j_scaled_final']]=[0,0,2.,1002.]
    write(root,df);s=audit(root,out)
    assert s['complete'] and s['invalid_records']==0 and s['valid_infeasible_records']==1

@pytest.mark.parametrize('field,value',[('j_scaled_final',np.nan),('j_scaled_final',np.inf),('runtime_sec',-2.),('runtime_sec',0.),('runtime_sec',np.inf),('v_cap_final',-1.),('v_tw_final','bad'),('accepted_final',2),('strict_duty_final',0),('seed',9),('method_id','WRONG'),('scenario_id','WRONG'),('eval_budget',999)])
def test_bad_record_fields(campaign,field,value):
    root,df,out=campaign;df[field]=df[field].astype(object);df.loc[0,field]=value;write(root,df.iloc[:4]);s=audit(root,out)
    assert s['invalid_records']==1 and not s['complete']

def test_missing_required_column_is_not_silently_skipped(campaign):
    root,df,out=campaign;write(root,df.iloc[:4].drop(columns='accepted_final'));s=audit(root,out)
    assert s['recorded_rows']==4 and s['invalid_records']==4
    issues=pd.read_csv(out/'validation_issues.csv');assert issues.issue.str.contains('missing_column:accepted_final').any()

def test_duplicates_and_unexpected(campaign):
    root,df,out=campaign;extra=df.iloc[:1].copy();extra['run_id']='UNEXPECTED';write(root,pd.concat([df,df.iloc[:1],extra]));s=audit(root,out)
    assert s['duplicate_rows']==2 and s['unexpected_rows']==1 and not s['complete']

def test_interrupted_checkpoint(campaign,tmp_path):
    root,df,out=campaign;execution=tmp_path/'execution';execution.mkdir()
    (execution/'attempt.json').write_text(json.dumps(dict(run_id=df.iloc[0].run_id,result_status='started',start_utc='2026-11-17T00:00:00Z')))
    s=audit(root,out,execution)
    assert s['incomplete_checkpoints']==1 and s['missing']==1080 and not s['complete']

def test_completed_checkpoint_metadata_must_match(campaign,tmp_path):
    root,df,out=campaign;execution=tmp_path/'execution';execution.mkdir()
    record=dict(run_id='WRONG',result_status='completed',result=df.iloc[0].to_dict(),wall_elapsed_s=2.,process_cpu_s=1.,start_utc='2026-11-17T00:00:00Z',end_utc='2026-11-17T00:00:02Z')
    (execution/'attempt.json').write_text(json.dumps(record));s=audit(root,out,execution)
    assert s['invalid_records']==1 and not s['complete']

def test_four_method_runtime_exclusion(campaign):
    root,df,out=campaign;df.loc[0,'runtime_sec']=20000.;df.loc[1,'runtime_sec']=9000.;write(root,df.iloc[:48]);s=audit(root,out)
    assert s['over_tolerance']==2 and s['paired_sensitivity_exclusions']==4 and s['sensitivity_retained_rows']==44
    excluded=pd.read_csv(out/'sensitivity_excluded_matched_cells.csv')
    assert set(excluded.method_id)=={'EDE','A1_NoSeed','A2_NoJDE','A3_NoLNS'}

def test_full_valid_with_deviation_requires_matched_handling(campaign):
    root,df,out=campaign;df.loc[0,'runtime_sec']=20000;write(root,df);s=audit(root,out)
    assert s['complete'] and not s['runtime_protocol_compliant']
    assert s['matched_runtime_handling_validated'] and s['scientific_closure_ready']

def test_json_and_csv_same_identity_are_duplicates(campaign,tmp_path):
    root,df,out=campaign;write(root,df.iloc[:1]);execution=tmp_path/'execution';execution.mkdir()
    r=df.iloc[0].to_dict();(execution/'one.json').write_text(json.dumps(dict(run_id=r['run_id'],result_status='completed',result=r,wall_elapsed_s=1.,process_cpu_s=.8,start_utc='2026-11-17T00:00:00Z',end_utc='2026-11-17T00:00:01Z')))
    s=audit(root,out,execution);assert s['duplicate_rows']==2 and not s['complete']

def test_incomplete_runtime_match_cannot_close(campaign):
    root,df,out=campaign;df.loc[0,'runtime_sec']=20000.;write(root,df.iloc[:3]);s=audit(root,out)
    assert not s['matched_runtime_handling_validated'] and not s['scientific_closure_ready']
