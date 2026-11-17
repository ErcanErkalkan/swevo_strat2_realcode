"""Operational wrapper; solver, budgets, seeds and immutable preregistration unchanged.

Atomic per-run records are the authoritative continuation ledger. No shared-host
parallelism; uncertain interrupted attempts require an explicit recorded decision.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,importlib.util,json,os,platform,socket,sys,time,uuid

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def classify(runtime,cap,tolerance=10):
    return 'over_tolerance' if runtime>cap+tolerance else ('above_nominal_within_tolerance' if runtime>cap else 'within_nominal')
def records(folder):
    found={}
    for p in sorted(folder.glob('*.json')):
        r=json.loads(p.read_text(encoding='utf-8'));rid=r['run_id']
        if rid in found: raise ValueError('Duplicate recorded run_id: '+rid)
        found[rid]=r
    return found
def recover(folder):
    for p in folder.glob('*.json'):
        r=json.loads(p.read_text())
        if r['result_status']=='started':
            r.update(result_status='interrupted',recovery_utc=utc(),end_utc=None,interruption_reason='No terminal record; prior process ended. Finish time and CPU unknown.')
            atomic(p,r)
def execute(plan,run,out,metadata):
    dest=out/(plan.run_id+'.json')
    if dest.exists(): raise ValueError('Existing attempt: '+plan.run_id)
    record=dict(metadata,run_id=plan.run_id,start_utc=utc(),end_utc=None,result_status='started',runtime_status='unknown',wall_elapsed_s=None,process_cpu_s=None)
    atomic(dest,record)
    wall=time.time(); mono=time.perf_counter();cpu=time.process_time()
    try:
        result=run(plan)
        raw=result.to_dict() if hasattr(result,'to_dict') else dict(result)
        if raw['run_id']!=plan.run_id: raise ValueError('Returned run identity differs')
        for name in ['instance_id','scenario_id','method_id','seed','tier','eval_budget']:
            if hasattr(plan,name) and raw.get(name)!=getattr(plan,name):raise ValueError('Returned plan metadata differs: '+name)
        record.update(result=raw,result_status='completed')
    except BaseException as exc:
        record.update(result_status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(end_utc=utc(),wall_elapsed_s=time.time()-wall,monotonic_elapsed_s=time.perf_counter()-mono,process_cpu_s=time.process_time()-cpu)
        elapsed=max(record['wall_elapsed_s'],float(record.get('result',{}).get('runtime_sec',0)))
        record['runtime_status']=classify(elapsed,plan.walltime_cap_s)
        record['nominal_cap_s']=plan.walltime_cap_s;record['logging_tolerance_s']=10
        atomic(dest,record)
    return record
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--instance');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    root=a.project_root.resolve(); campaign=a.campaign.resolve();out=a.output.resolve()
    # One fixed TCP port per host protects all C15 wrappers across copies/shards.
    guard=socket.socket();guard.bind(('127.0.0.1',49615));guard.listen(1)
    spec=importlib.util.spec_from_file_location('locked_runner',campaign/'run_confirmatory_ablation.py');locked=importlib.util.module_from_spec(spec);spec.loader.exec_module(locked)
    lock=campaign/'confirmatory_ablation_source_lock.json';manifest=campaign/'confirmatory_ablation_manifest_1080_cli.csv';selection=campaign/'representative_instance_selection.csv'
    locked.verify_source_lock(root,lock);locked.verify_benchmarks(root,selection)
    config=json.loads(lock.read_text())
    assert sha(manifest)==config['manifest_sha256'] and sha(selection)==config['selection_sha256'],'Manifest/selection hash drift'
    os.environ['SWEVO_REQUIRE_REAL_BENCHMARKS']='1';sys.path.insert(0,str(root/'src'))
    from swevo_suite.manifest import load_manifest
    from swevo_suite.comparators import REGISTRY
    plans=load_manifest(manifest);expected={p.run_id for p in plans}
    assert len(plans)==1080 and len(expected)==1080
    # Historical rows remain byte-preserved; duplicate identities fail closed.
    historical={}
    for p in sorted((campaign/'results').glob('*.csv')):
        with p.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                rid=r['run_id']
                if rid in historical or rid not in expected: raise ValueError('Duplicate/unexpected historical identity '+rid)
                historical[rid]=r
    out.mkdir(parents=True,exist_ok=True);recover(out);prior=records(out)
    if set(prior)-expected or set(prior)&set(historical): raise ValueError('Unexpected/duplicate attempt identity')
    if any(r['result_status'] in ['interrupted','failed'] for r in prior.values()): raise RuntimeError('Interrupted/failed attempts require a separately recorded matched-cell decision; no automatic selective rerun.')
    todo=[p for p in plans if p.run_id not in historical and p.run_id not in prior and (not a.instance or p.instance_id==a.instance)]
    print(json.dumps(dict(preflight='PASS',historical=len(historical),recorded=len(prior),todo=len(todo),remaining_nominal_hours=sum(p.walltime_cap_s for p in todo)/3600),indent=2),flush=True)
    if a.dry_run:return
    metadata=dict(host=platform.node(),platform=platform.platform(),python=sys.version,environment_sha256=hashlib.sha256(json.dumps(sorted((d.metadata['Name'],d.version) for d in __import__('importlib.metadata',fromlist=['distributions']).distributions())).encode()).hexdigest(),manifest_sha256=sha(manifest),source_lock_sha256=sha(lock),runner_sha256=sha(Path(__file__)),solver_source_hashes=config['files'])
    # Prevent automatic Windows system sleep only while this process is active;
    # restore the thread setting even when a run is interrupted. No power plan changes.
    keepawake=None
    if os.name=='nt':
        import ctypes
        keepawake=ctypes.windll.kernel32.SetThreadExecutionState
        if not keepawake(0x80000001):raise OSError('Cannot request temporary system-awake execution')
    try:
        for plan in todo:
            locked.verify_source_lock(root,lock)
            def run(p):
                cls=REGISTRY[p.method_id];method=cls(p.method_id) if p.method_id in locked.ABLATIONS else cls();return method.run(p)
            r=execute(plan,run,out,metadata); print(plan.run_id,r['result_status'],r['runtime_status'],flush=True)
    finally:
        if keepawake:keepawake(0x80000000)
if __name__=='__main__':main()
