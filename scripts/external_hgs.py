"""Original PyVRP 0.12.2 HGS adapter on the explicitly narrowed common VRPTW."""
from pathlib import Path
import sys,time,math,copy,json,argparse,socket,os
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT.parent/'deps_hgs'))
from swevo_suite import solver
from swevo_suite.benchmark import build_problem,load_scenarios
from swevo_suite.models import Route,Shift,VehicleType
from swevo_suite.comparators.ede import EDEComparator
from swevo_suite.manifest import load_manifest
from swevo_suite.c17_telemetry import Telemetry,instrument
from swevo_suite.c17_verifier import verify,serialize_solution
from run_ablation import atomic,utc
from c17_metadata import metadata
from pyvrp import Model

SCALE=1000


def narrow(problem):
    p=copy.deepcopy(problem);v=p.vehicle_types['ICE']
    p.vehicle_types={name:replace(v,name=name,fixed_cost=0.) for name in ['ICE','HEV','EV']}
    p.shifts=[Shift(1,0,p.shifts[0].end)];p.shift_factors={1:1.};p.objective_weights=(1.,0.,0.);p.eta=0.
    return p


def distance(a,b):return round(math.hypot(a[0]-b[0],a[1]-b[1])*SCALE)/SCALE


def reference(problem):
    return sum(2*distance(problem.depot_xy,(c.x,c.y)) for c in problem.customers.values())


def make_model(problem):
    m=Model();horizon=round(problem.shifts[0].end*SCALE)
    depot=m.add_depot(*problem.depot_xy,tw_early=0,tw_late=horizon)
    ids=sorted(problem.customers);locations=[depot];coords=[problem.depot_xy]
    for cid in ids:
        c=problem.customers[cid];coords.append((c.x,c.y))
        locations.append(m.add_client(c.x,c.y,delivery=round(c.demand),service_duration=round(c.service*SCALE),tw_early=round(c.tw_start*SCALE),tw_late=round(c.tw_end*SCALE)))
    m.add_vehicle_type(num_available=len(ids),capacity=round(problem.vehicle_types['ICE'].capacity),tw_early=0,tw_late=horizon,max_reloads=0)
    for i,a in enumerate(locations):
        for j,b in enumerate(locations):
            v=round(distance(coords[i],coords[j])*SCALE);m.add_edge(a,b,distance=v,duration=v)
    return m,ids


class ObserveStop:
    def __init__(self,start,cap,ref):self.start=start;self.cap=cap;self.ref=ref;self.events=[]
    def __call__(self,cost):
        t=time.perf_counter()-self.start
        if t<=self.cap and math.isfinite(cost) and cost<2**62:
            val=float(cost)/SCALE/self.ref
            if not self.events or val<self.events[-1][1]:self.events.append((t,val))
        return t>=self.cap


def execute(plan,method):
    total_start=time.perf_counter();p=narrow(build_problem(plan,load_scenarios(),require_real=True));ref=reference(p)
    model,ids=make_model(p) if method=='PYVRP_HGS' else (None,None)
    preparation=time.perf_counter()-total_start
    start=time.perf_counter();original_distance=solver.euclidean;solver.euclidean=distance
    try:
        if method=='PYVRP_HGS':
            stop=ObserveStop(start,plan.walltime_cap_s,ref);res=model.solve(stop,seed=plan.seed,display=False,collect_stats=False)
            elapsed=time.perf_counter()-start
            routes=[Route(1,'ICE',[ids[i-1] for i in r.visits()]) for r in res.best.routes()]
            best=solver.evaluate_solution(p,routes,(ref,1,1))
            telemetry={'initial_score':None,'initial_score_valid':False,'first_feasible_s':None,'first_feasible_valid':False,
                'full_candidate_evaluations':None,'evaluation_count_valid':False,'repair_attempts':None,'repair_successes':None,'repair_count_valid':False,
                'reason':'C++ nested candidate events not exposed; public stopping hook gives observation bounds only',
                'observed_first_feasible_upper_bound_s':stop.events[0][0] if stop.events else None,
                'no_feasible_observed_within_budget':not stop.events,'search_elapsed_s':elapsed,'preparation_s':preparation,
                'nominal_budget_s':plan.walltime_cap_s,'budget_deviation_s':elapsed-plan.walltime_cap_s,'anytime':[]}
            for f in [0,.1,.25,.5,.75,1]:
                vals=[x for t,x in stop.events if t<=f*plan.walltime_cap_s]
                telemetry['anytime'].append({'time_s':f*plan.walltime_cap_s,'best_feasible_score':min(vals) if vals else None,'valid':bool(vals)})
        else:
            e=Telemetry(p,plan.walltime_cap_s)
            with instrument(e):best,stats,archive=solver.jde_evolve(p,EDEComparator().config_for_plan(plan))
            elapsed=time.perf_counter()-e.start
            # EDE internal reference differs; convert distance-only scores to the
            # common deterministic singleton-distance scalar for both methods.
            factor=best.references[0]/ref
            telemetry=e.report(elapsed,preparation,time.perf_counter()-total_start)
            if telemetry['initial_score'] is not None:telemetry['initial_score']*=factor
            for point in telemetry['anytime']:
                if point['best_feasible_score'] is not None:point['best_feasible_score']*=factor
            best=solver.evaluate_solution(p,best.routes,(ref,1,1))
        raw=serialize_solution(best);audit=verify(p,raw,(ref,1,1),{'score':best.score,'cost':best.cost,'energy':best.energy,'co2':best.co2},arc_rounding_scale=SCALE)
        telemetry['total_elapsed_s']=time.perf_counter()-total_start
        return {'cohort':'NEW_NARROW_EXTERNAL','run_id':plan.run_id,'method':method,'plan':asdict(plan),'common_reference_distance':ref,
            'final':{'score':best.score,'cost':best.cost,'energy':best.energy,'co2':best.co2,'accepted':best.accepted},
            'routes':raw,'independent_validation':audit,'telemetry':telemetry,'arc_rounding_scale':SCALE}
    finally:solver.euclidean=original_distance


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int);a=ap.parse_args()
    lock=socket.socket();lock.bind(('127.0.0.1',49615));lock.listen(1)
    meta=metadata()
    for plan in load_manifest(a.manifest)[:a.limit]:
        path=a.output/(plan.run_id+'.json')
        if path.exists():
            if json.loads(path.read_text())['status']=='completed':continue
            raise RuntimeError('Unresolved prior attempt '+str(path))
        import hashlib
        record={'run_id':plan.run_id,'status':'started','start_utc':utc(),'provenance':meta,'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()};atomic(path,record)
        try:record.update(status='completed',result=execute(plan,plan.method_id))
        except BaseException as exc:record.update(status='failed',error=repr(exc));raise
        finally:record['end_utc']=utc();atomic(path,record)
        print(plan.run_id,record['result']['final'],record['result']['independent_validation']['valid'],flush=True)
        from c17_monitor import refresh
        refresh(ROOT)
if __name__=='__main__':main()
