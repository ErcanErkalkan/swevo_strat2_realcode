"""Read all raw result shards; never overwrite preregistration or raw results."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
from analyze_c15_primary import ids,sha

def audit(campaign,out,execution=None):
    out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(campaign/'confirmatory_ablation_manifest_1080_cli.csv');manifest['run_id']=ids(manifest)
    assert len(manifest)==1080 and manifest.run_id.nunique()==1080
    files=sorted(p for p in campaign.rglob('*.csv') if 'results' in p.name and not p.name.startswith('development'))
    frames=[]
    for p in files:
        df=pd.read_csv(p)
        if 'run_id' not in df: continue
        df['source_file']=p.relative_to(campaign).as_posix();frames.append(df)
    attempts=[]
    if execution:
        for p in sorted(execution.glob('*.json')):
            record=json.loads(p.read_text());attempts.append(record)
            if record['result_status']=='completed':
                row=dict(record['result']);row['source_file']=str(p)
                row['runtime_sec']=max(float(row['runtime_sec']),float(record['wall_elapsed_s']))
                frames.append(pd.DataFrame([row]))
    d=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['run_id'])
    d.to_csv(out/'c15_ablation_combined_observed.csv',index=False)
    (out/'c15_ablation_execution_attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
    got=set(d.run_id);expected=set(manifest.run_id)
    missing=manifest[~manifest.run_id.isin(got)]
    duplicates=d[d.run_id.duplicated(keep=False)]
    unexpected=d[~d.run_id.isin(expected)]
    matches=d.merge(manifest[['run_id','walltime_cap_s']],on='run_id',validate='many_to_one')
    matches['over_tolerance']=matches.runtime_sec>matches.walltime_cap_s+10
    over=matches[matches.over_tolerance]
    bad=over[['instance_id','scenario_id','seed']].drop_duplicates()
    paired=manifest.merge(bad,on=['instance_id','scenario_id','seed'])
    paired=paired.merge(d[['run_id','runtime_sec']],on='run_id',how='left')
    for name,frame in [('missing',missing),('duplicates',duplicates),('unexpected',unexpected),('runtime_deviations',over),('sensitivity_excluded_matched_cells',paired)]: frame.to_csv(out/f'c15_ablation_{name}.csv',index=False)
    status=dict(schema='c15-ablation-status-v1',expected=1080,recorded_rows=len(d),unique_run_ids=len(got),matched_unique=len(got&expected),missing=len(missing),duplicate_rows=len(duplicates),unexpected_rows=len(unexpected),over_tolerance=len(over),paired_sensitivity_exclusions=len(paired),remaining_nominal_hours=float(missing.walltime_cap_s.sum()/3600),complete=len(d)==1080 and len(got)==1080 and not len(unexpected) and not len(duplicates),root_cause='belirlenemedi',raw_files={p.relative_to(campaign).as_posix():sha(p) for p in files})
    (out/'c15_ablation_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2));return status
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--execution',type=Path);a=ap.parse_args();audit(a.campaign,a.out,a.execution)
