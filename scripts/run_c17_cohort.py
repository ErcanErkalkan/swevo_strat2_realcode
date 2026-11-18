"""Separate instrumented cohort executor; atomic started/terminal records."""
from pathlib import Path
import sys,os,json,time,argparse,hashlib,random,socket
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swevo_suite import solver
from swevo_suite.benchmark import build_problem,load_scenarios
from swevo_suite.comparators import REGISTRY
from swevo_suite.manifest import load_manifest
from swevo_suite.c17_telemetry import Telemetry,instrument
from swevo_suite.c17_verifier import verify,serialize_solution
from run_ablation import atomic,utc
from c17_metadata import metadata


def run(plan,selection='boundary'):
    t0=time.perf_counter();cls=REGISTRY[plan.method_id]
    method=cls(plan.method_id) if plan.method_id.startswith('A') and plan.method_id not in ['ALNS_MS'] else cls()
    problem=build_problem(plan,load_scenarios(),require_real=True);cfg=method.config_for_plan(plan)
    preparation=time.perf_counter()-t0
    events=Telemetry(problem,plan.walltime_cap_s);old=solver.C17_REMOVAL_SELECTION
    solver.C17_REMOVAL_SELECTION=selection
    kernels={'EDE':solver.jde_evolve,'StdDE':solver.jde_evolve,'ALNS_MS':solver.alns_search,'HGS_MS':solver.hgs_search,'ILS_MS':solver.ils_search}
    try:
        with instrument(events):best,stats,archive=kernels.get(plan.method_id,solver.jde_evolve)(problem,cfg)
        search=time.perf_counter()-events.start
    finally:solver.C17_REMOVAL_SELECTION=old
    # Capture the kernel's end before route audit and return/serialization work.
    routes=serialize_solution(best);validation=verify(problem,routes,best.references,asdict(best))
    telemetry=events.report(search,preparation,time.perf_counter()-t0)
    result={'run_id':plan.run_id,'cohort':'C17_INSTRUMENTED','plan':asdict(plan),'selection':selection,
            'telemetry':telemetry,'final':{k:getattr(best,k) for k in ['score','cost','energy','co2','accepted','strict_duty','overtime_sum']},
            'routes':routes,'references':best.references,'independent_validation':validation,
            'within_budget_routes':serialize_solution(events.best) if events.best else None}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',choices=['boundary','random'],default='boundary');p.add_argument('--selection-from-manifest',action='store_true');p.add_argument('--limit',type=int);a=p.parse_args()
    lock=socket.socket();lock.bind(('127.0.0.1',49615));lock.listen(1)
    os.environ['SWEVO_REQUIRE_REAL_BENCHMARKS']='1'
    plans=load_manifest(a.manifest)
    import csv
    with a.manifest.open() as f:design=list(csv.DictReader(f))
    meta=metadata()
    for plan,design_row in list(zip(plans,design))[:a.limit]:
        selection=design_row['selection'] if a.selection_from_manifest else a.selection
        path=a.output/(plan.run_id+'__'+selection+'.json')
        if path.exists():
            prior=json.loads(path.read_text())
            if prior['status']=='completed':continue
            raise RuntimeError('Unresolved prior attempt '+str(path))
        record={'run_id':plan.run_id,'status':'started','start_utc':utc(),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'provenance':meta};atomic(path,record)
        try:record.update(status='completed',result=run(plan,selection))
        except BaseException as exc:record.update(status='failed',error=repr(exc));raise
        finally:record['end_utc']=utc();atomic(path,record)
        print(plan.run_id,record['status'],flush=True)
        from c17_monitor import refresh
        refresh(ROOT)
if __name__=='__main__':main()
