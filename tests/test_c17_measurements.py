import pytest
from swevo_suite.models import Customer,Shift,VehicleType,ProblemInstance,Route
from swevo_suite.solver import evaluate_solution
from swevo_suite.c17_verifier import verify,serialize_solution
from swevo_suite.c17_telemetry import Telemetry


def fixture(end=100,tw=100,cap=10):
    return ProblemInstance('fixture','test','test',1,{1:Customer(1,3,4,2,0,tw,0,'z')},[Shift(1,0,end)],(0,0),{'v':VehicleType('v',cap,7,2,1,3)},{1:2},{'z':4})


def test_independent_values_and_tamper_detection():
    p=fixture();s=evaluate_solution(p,[Route(1,'v',[1])],(17,30,720));r=serialize_solution(s)
    audit=verify(p,r,s.references,{'score':s.score})
    assert audit['valid'] and audit['recomputed']['co2']==pytest.approx(720)
    r[0]['service_starts']={1:6};assert not verify(p,r)['valid']


@pytest.mark.parametrize('p,r,reason',[(fixture(cap=1),[Route(1,'v',[1])],'capacity'),(fixture(tw=4),[Route(1,'v',[1])],'time_window'),(fixture(end=9),[Route(1,'v',[1])],'depot_return'),(fixture(),[],'coverage'),(fixture(),[Route(1,'v',[1,1])],'coverage')])
def test_violations_detected(p,r,reason):
    a=verify(p,[{'shift_id':x.shift_id,'vehicle_type':x.vehicle_type,'customers':x.customers} for x in r])
    assert not a['valid'] and any(reason in x for x in a['errors'])


def test_missing_measurement_is_not_final_fallback_and_timeout_remains():
    now=[0.];p=fixture();e=Telemetry(p,1,clock=lambda:now[0]);now[0]=2
    s=evaluate_solution(p,[Route(1,'v',[1])],(17,30,720));e.observe(s)
    r=e.report(12,0,12)
    assert r['initial_score']==1 and r['first_feasible_s'] is None
    assert r['no_feasible_within_budget'] and r['timing_status']=='over_tolerance'
    assert all(x['best_feasible_score'] is None for x in r['anytime'])
    blank=Telemetry(p,1,clock=lambda:0).report(1,0,1)
    assert blank['initial_score'] is None and not blank['initial_score_valid']
    assert blank['repair_attempts']==0 and blank['repair_count_valid']


def test_true_zero_first_solution_time_is_valid():
    p=fixture();e=Telemetry(p,1,clock=lambda:0)
    e.observe(evaluate_solution(p,[Route(1,'v',[1])],(17,30,720)))
    r=e.report(1,0,1);assert r['first_feasible_s']==0 and r['first_feasible_valid']
