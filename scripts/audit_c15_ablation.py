"""Validate coverage, scientific records and timing independently; raw inputs are read-only."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,math
import pandas as pd
METHODS={'EDE','A1_NoSeed','A2_NoJDE','A3_NoLNS'}
META=['phase','benchmark_family','customer_count','instance_id','structure_class','scenario_id','method_id','seed','tier','eval_budget']
INT_META={'customer_count','seed','eval_budget'}
CONSTRAINTS=['v_cap_final','v_tw_final','v_shift_final','overtime_sum_final','overtime_ratio_sum_final']
NUMERIC=['j_scaled_final','runtime_sec',*CONSTRAINTS,'accepted_final','strict_duty_final']
REQUIRED=['run_id',*META,*NUMERIC]
CELL=['instance_id','scenario_id','seed']
TOL=10.
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def number(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError,OverflowError):return None
def identifier(row):return f"{row['instance_id']}__{row['scenario_id']}__{row['method_id']}__seed{int(row['seed']):02d}"
def load_design(path):
    manifest=pd.read_csv(path)
    missing=set(META+['walltime_cap_s'])-set(manifest)
    if missing:raise ValueError('Manifest missing columns: '+','.join(sorted(missing)))
    if manifest[META+['walltime_cap_s']].isna().any().any():raise ValueError('Blank manifest field')
    for key in INT_META|{'walltime_cap_s'}:
        vals=pd.to_numeric(manifest[key],errors='coerce')
        if vals.isna().any() or not vals.map(math.isfinite).all() or not vals.gt(0).all() or not vals.eq(vals.round()).all():raise ValueError('Invalid manifest integer '+key)
        manifest[key]=vals.astype(int)
    manifest['run_id']=manifest.apply(identifier,axis=1)
    grid=manifest.groupby(CELL).method_id.agg(set)
    caps=manifest.tier.map({'small':1800,'medium':3600,'large':7200})
    if len(manifest)!=1080 or manifest.run_id.nunique()!=1080 or manifest.instance_id.nunique()!=9 or manifest.scenario_id.nunique()!=3 or set(manifest.method_id)!=METHODS or set(manifest.seed)!=set(range(1,11)) or not grid.map(lambda x:x==METHODS).all() or not manifest.walltime_cap_s.eq(caps).all():raise ValueError('Manifest is not the preregistered 9 x 3 x 4 x 10 grid and tier budgets')
    return manifest
def checkpoint_errors(record,row):
    errors=[]
    if record.get('run_id')!=row.get('run_id'):errors.append('checkpoint_run_id_mismatch')
    for field in ['wall_elapsed_s','process_cpu_s']:
        val=number(record.get(field))
        if val is None or val<0 or (field=='wall_elapsed_s' and val==0):errors.append('invalid_checkpoint:'+field)
    stamps=[]
    for field in ['start_utc','end_utc']:
        try:
            stamp=datetime.datetime.fromisoformat(str(record[field]).replace('Z','+00:00'))
            if stamp.utcoffset()!=datetime.timedelta(0):raise ValueError()
            stamps.append(stamp)
        except (KeyError,ValueError,TypeError):errors.append('invalid_checkpoint:'+field)
    if len(stamps)==2 and stamps[1]<stamps[0]:errors.append('checkpoint_end_before_start')
    return errors
def validate_row(row,plan):
    issues=list(row.pop('_checkpoint_issues',[]))
    for field in REQUIRED:
        if field not in row:issues.append('missing_column:'+field)
        elif row[field] is None or str(row[field]).strip()=='':issues.append('blank_field:'+field)
    if plan is None:issues.append('unexpected_run_id')
    else:
        for field in META:
            if field not in row:continue
            if field in INT_META:
                val=number(row[field])
                if val is None or val!=int(val) or int(val)!=int(plan[field]):issues.append('manifest_mismatch:'+field)
            elif str(row[field])!=str(plan[field]):issues.append('manifest_mismatch:'+field)
    parsed={field:number(row.get(field)) for field in NUMERIC}
    for field,value in parsed.items():
        if value is None:issues.append('nonfinite_or_invalid:'+field)
        elif field=='runtime_sec' and value<=0:issues.append('nonpositive:runtime_sec')
        elif field in CONSTRAINTS and value<0:issues.append('negative:'+field)
        elif field in ['accepted_final','strict_duty_final'] and value not in [0.,1.]:issues.append('invalid_flag:'+field)
    if all(parsed[k] is not None for k in CONSTRAINTS+['accepted_final','strict_duty_final']):
        hard=all(parsed[k]<=1e-9 for k in ['v_cap_final','v_tw_final','v_shift_final'])
        if parsed['accepted_final']!=int(hard):issues.append('inconsistent_feasibility:accepted_final')
        if parsed['strict_duty_final']!=int(hard and parsed['overtime_sum_final']<=1e-9 and parsed['overtime_ratio_sum_final']<=1e-9):issues.append('inconsistent_feasibility:strict_duty_final')
    return sorted(set(issues)),parsed
def audit(campaign,out,execution=None):
    campaign=Path(campaign);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest_path=campaign/'confirmatory_ablation_manifest_1080_cli.csv';manifest=load_design(manifest_path)
    lookup=manifest.set_index('run_id').to_dict('index');expected=set(lookup)
    rows=[];file_issues=[];incomplete=[];sources={}
    files=sorted(p for p in campaign.rglob('*.csv') if 'results' in p.name and not p.name.startswith('development'))
    for path in files:
        name=path.relative_to(campaign).as_posix();sources[name]=sha(path)
        try:
            with path.open(encoding='utf-8-sig',newline='') as handle:
                reader=csv.DictReader(handle)
                if not reader.fieldnames:file_issues.append(dict(source_file=name,issue='empty_file_without_header'))
                for i,row in enumerate(reader,2):rows.append(dict(row,source_file=name,source_row=i))
        except (OSError,csv.Error,UnicodeError) as exc:file_issues.append(dict(source_file=name,issue=type(exc).__name__))
    if execution:
        for path in sorted(Path(execution).glob('*.json')):
            name='execution/'+path.name;sources[name]=sha(path)
            try:
                record=json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(record,dict):raise ValueError('not an object')
            except (ValueError,OSError,UnicodeError) as exc:
                file_issues.append(dict(source_file=name,issue='invalid_checkpoint_json:'+type(exc).__name__));continue
            if record.get('result_status')!='completed':
                incomplete.append(dict(run_id=record.get('run_id',''),source_file=name,result_status=record.get('result_status','missing')));continue
            row=record.get('result')
            if not isinstance(row,dict):file_issues.append(dict(source_file=name,issue='completed_checkpoint_without_result'));continue
            row=dict(row);row.update(source_file=name,source_row=1,wall_elapsed_s=record.get('wall_elapsed_s'),process_cpu_s=record.get('process_cpu_s'),start_utc=record.get('start_utc'),end_utc=record.get('end_utc'))
            row['_checkpoint_issues']=checkpoint_errors(record,row);rows.append(row)
    issues=[];validated=[];invalid=[]
    for row in rows:
        rid=str(row.get('run_id','')).strip();row['run_id']=rid
        errors,parsed=validate_row(row,lookup.get(rid))
        for issue in errors:issues.append(dict(run_id=rid,source_file=row['source_file'],source_row=row['source_row'],issue=issue))
        if errors:invalid.append(row)
        else:
            valid=dict(row);valid.update(parsed);valid.update({key:int(lookup[rid][key]) for key in INT_META})
            valid['effective_wall_sec']=max(parsed['runtime_sec'],number(row.get('wall_elapsed_s')) or parsed['runtime_sec'])
            valid['walltime_cap_s']=lookup[rid]['walltime_cap_s'];validated.append(valid)
    columns=list(dict.fromkeys(REQUIRED+['source_file','source_row','wall_elapsed_s','process_cpu_s','start_utc','end_utc']))
    all_rows=pd.DataFrame(rows).reindex(columns=list(dict.fromkeys(columns+[key for row in rows for key in row])))
    valid_df=pd.DataFrame(validated).reindex(columns=columns+['effective_wall_sec','walltime_cap_s'])
    got=set(all_rows.run_id.dropna())-{''};missing=manifest[~manifest.run_id.isin(got)]
    dup_mask=all_rows.run_id.duplicated(keep=False)&all_rows.run_id.ne('');duplicates=all_rows[dup_mask]
    unexpected=all_rows[~all_rows.run_id.isin(expected)]
    over=valid_df[valid_df.effective_wall_sec.gt(valid_df.walltime_cap_s+TOL)]
    bad=over[CELL].drop_duplicates()
    if bad.empty:
        paired=manifest.iloc[:0].copy();paired['runtime_sec']=pd.Series(dtype=float);paired['effective_wall_sec']=pd.Series(dtype=float)
    else:paired=manifest.merge(bad,on=CELL).merge(valid_df[['run_id','runtime_sec','effective_wall_sec']],on='run_id',how='left')
    matched_ok=True
    for _,cell in bad.iterrows():
        group=valid_df
        for name in CELL:group=group[group[name].astype(str)==str(cell[name])]
        if len(group)!=4 or group.run_id.nunique()!=4 or set(group.method_id)!=METHODS:matched_ok=False
    bad_keys=set(map(tuple,bad.astype(str).to_numpy()))
    mask=[tuple(str(row[k]) for k in CELL) not in bad_keys for _,row in valid_df.iterrows()]
    retained=valid_df.loc[mask] if len(valid_df) else valid_df.copy()
    coverage=len(rows)==1080 and len(got&expected)==1080 and len(duplicates)==0 and len(unexpected)==0 and len(missing)==0
    validation=not invalid and not file_issues and not incomplete;complete=bool(coverage and validation)
    outputs=[('combined_observed',all_rows),('validated_records',valid_df),('missing',missing),('duplicates',duplicates),('unexpected',unexpected),('invalid_records',pd.DataFrame(invalid).reindex(columns=columns)),('validation_issues',pd.DataFrame(issues,columns=['run_id','source_file','source_row','issue'])),('file_issues',pd.DataFrame(file_issues,columns=['source_file','issue'])),('incomplete_checkpoints',pd.DataFrame(incomplete,columns=['run_id','source_file','result_status'])),('runtime_deviations',over),('sensitivity_excluded_matched_cells',paired),('sensitivity_retained_records',retained)]
    for name,frame in outputs:frame.to_csv(out/f'c15_ablation_{name}.csv',index=False)
    status=dict(schema='c15.1-ablation-status-v2',expected=1080,recorded_rows=len(rows),unique_run_ids=len(got),matched_unique=len(got&expected),missing=len(missing),duplicate_rows=len(duplicates),unexpected_rows=len(unexpected),invalid_records=len(invalid),valid_records=len(valid_df),valid_infeasible_records=int(valid_df.accepted_final.eq(0).sum()),file_errors=len(file_issues),incomplete_checkpoints=len(incomplete),coverage_complete=bool(coverage),record_validation_passed=bool(validation),complete=complete,over_tolerance=len(over),runtime_tolerance_s=TOL,runtime_protocol_compliant=bool(len(over)==0 and validation),runtime_protocol_status='not_assessed' if not len(valid_df) else ('deviations_present' if len(over) else 'within_tolerance'),paired_sensitivity_exclusions=len(paired),sensitivity_retained_rows=len(retained),matched_runtime_handling_validated=bool(matched_ok),scientific_closure_ready=bool(complete and matched_ok),remaining_nominal_hours=float(missing.walltime_cap_s.sum()/3600),root_cause='belirlenemedi',manifest_sha256=sha(manifest_path),raw_files=sources)
    (out/'c15_ablation_status.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(status,indent=2));return status
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--execution',type=Path);a=ap.parse_args();audit(a.campaign,a.out,a.execution)
