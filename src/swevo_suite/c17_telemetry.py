"""C17-only event instrumentation. Never load in the frozen ablation process.

Initial score is the first full-coverage evaluated candidate, not a retrospective
best-in-population estimate. Full and partial candidate evaluations are distinct.
All clocks are monotonic. Missing endpoints remain None with a validity flag.
"""
from contextlib import contextmanager
import math,time


class Telemetry:
    def __init__(self, problem, cap, clock=time.perf_counter):
        self.problem=problem;self.cap=float(cap);self.clock=clock;self.start=clock()
        self.initial_score=None;self.first_feasible_s=None;self.evaluations=0
        self.partial_evaluations=0;self.repair_attempts=0;self.repair_successes=0
        self.improvements=[];self.best=None

    def observe(self,sol):
        t=self.clock()-self.start
        served=[c for r in sol.routes for c in r.customers]
        complete=len(served)==len(self.problem.customers) and set(served)==set(self.problem.customers)
        if not complete:self.partial_evaluations+=1;return
        self.evaluations+=1
        if self.initial_score is None and math.isfinite(sol.score):self.initial_score=float(sol.score)
        # Never backfill a budget endpoint from a late return.
        if sol.accepted and t<=self.cap:
            if self.first_feasible_s is None:self.first_feasible_s=t
            if self.best is None or sol.score<self.best.score:
                self.best=sol.copy();self.improvements.append((t,float(sol.score)))

    def report(self,search_elapsed_s,preparation_s,total_s):
        points=[]
        for f in [0,.1,.25,.5,.75,1]:
            at=self.cap*f;known=[score for t,score in self.improvements if t<=at]
            points.append({'time_s':at,'best_feasible_score':min(known) if known else None,'valid':bool(known)})
        return {'schema':'c17-events-v1','initial_score':self.initial_score,'initial_score_valid':self.initial_score is not None,
            'initial_definition':'first_full_coverage_candidate_evaluated',
            'first_feasible_s':self.first_feasible_s,'first_feasible_valid':self.first_feasible_s is not None,
            'no_feasible_within_budget':self.first_feasible_s is None,'full_candidate_evaluations':self.evaluations,
            'partial_candidate_evaluations':self.partial_evaluations,'evaluation_count_valid':True,
            'repair_attempts':self.repair_attempts,'repair_successes':self.repair_successes,'repair_count_valid':True,
            'repair_definition':'bounded_repair invocation on infeasible input before deadline; success is feasible return',
            'search_elapsed_s':search_elapsed_s,'preparation_s':preparation_s,'total_elapsed_s':total_s,
            'nominal_budget_s':self.cap,'budget_deviation_s':search_elapsed_s-self.cap,
            'timing_status':'over_tolerance' if search_elapsed_s>self.cap+10 else 'above_nominal' if search_elapsed_s>self.cap else 'within_nominal',
            'anytime':points,'improvement_events':self.improvements}


@contextmanager
def instrument(events):
    from . import solver
    names=['evaluate_solution','_assemble_solution_from_route_update','bounded_repair']
    originals={name:getattr(solver,name) for name in names}
    def evaluated(fn):
        def wrapped(*a,**kw):
            sol=fn(*a,**kw);events.observe(sol);return sol
        return wrapped
    def repair(*a,**kw):
        inp=a[1];deadline=kw.get('deadline')
        attempt=not inp.accepted and (deadline is None or time.perf_counter()<deadline)
        if attempt:events.repair_attempts+=1
        sol=originals['bounded_repair'](*a,**kw)
        if attempt and sol.accepted:events.repair_successes+=1
        return sol
    solver.evaluate_solution=evaluated(originals['evaluate_solution'])
    solver._assemble_solution_from_route_update=evaluated(originals['_assemble_solution_from_route_update'])
    solver.bounded_repair=repair
    try:yield
    finally:
        for name,fn in originals.items():setattr(solver,name,fn)
