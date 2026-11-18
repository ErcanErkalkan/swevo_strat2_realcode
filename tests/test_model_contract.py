"""Small analytical fixtures test evaluator contracts, not optimizer outcomes."""
import pytest
from swevo_suite.models import Customer,Shift,VehicleType,ProblemInstance,Route
from swevo_suite.solver import evaluate_solution

def problem(end=100,capacity=10,tw_end=100):
    # 3-4-5 triangle, one two-unit delivery: distance=10, energy=20+10=30.
    return ProblemInstance(name='ANALYTICAL_TEST',family='test',structure_class='test',customer_count=1,
        customers={1:Customer(1,3,4,2,0,tw_end,0,'z')},shifts=[Shift(1,0,end)],depot_xy=(0,0),
        vehicle_types={'v':VehicleType('v',capacity,7,2,1,3)},shift_factors={1:2},zone_factors={'z':4},eta=.05)

def test_analytic_route_energy_emission_and_normalized_objective():
    s=evaluate_solution(problem(),[Route(1,'v',[1])],(17,30,720))
    assert (s.cost,s.energy,s.co2)==pytest.approx((17,30,720))
    assert s.score==pytest.approx(1) and s.accepted and s.strict_duty
    assert s.routes[0].metrics.return_time==10

@pytest.mark.parametrize('routes', [[],[Route(1,'v',[1,1])]])
def test_missing_or_duplicate_customer_invalidates_solution(routes):
    s=evaluate_solution(problem(),routes,(17,30,720))
    assert not s.accepted and not s.strict_duty and s.v_shift>0 and s.score>=1000

@pytest.mark.parametrize('p,field', [(problem(capacity=1),'v_cap'),(problem(tw_end=4),'v_tw'),(problem(end=9),'v_shift')])
def test_hard_constraints_are_enforced(p,field):
    s=evaluate_solution(p,[Route(1,'v',[1])],(17,30,720))
    assert getattr(s,field)>0 and not s.accepted

def test_bounded_overtime_is_feasible_but_not_strict():
    s=evaluate_solution(problem(end=9.8),[Route(1,'v',[1])],(17,30,720))
    assert s.accepted and not s.strict_duty
    assert s.overtime_sum==pytest.approx(.2) and s.score>1
