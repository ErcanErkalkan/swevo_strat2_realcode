"""Independent route audit: no solver evaluator or acceptance function imported."""
from collections import Counter
from math import hypot,isclose,isfinite


def verify(problem,routes,references=None,reported=None,arc_rounding_scale=None):
    errors=[];seen=[];cost=energy=co2=ot_sum=ot_ratio=0.;route_info=[]
    shift_map={s.idx:s for s in problem.shifts}
    for index,r in enumerate(routes):
        customers=list(r['customers']);seen.extend(customers)
        if r['shift_id'] not in shift_map or r['vehicle_type'] not in problem.vehicle_types:
            errors.append(f'route:{index}:unknown_shift_or_vehicle');continue
        if any(c not in problem.customers for c in customers):
            errors.append(f'route:{index}:unknown_customer');continue
        if not customers:continue
        s=shift_map[r['shift_id']];v=problem.vehicle_types[r['vehicle_type']]
        if s.end<=s.start:errors.append(f'route:{index}:invalid_shift');continue
        load=sum(problem.customers[c].demand for c in customers)
        if load>v.capacity+1e-9:errors.append(f'route:{index}:capacity')
        t=s.start;xy=problem.depot_xy;starts={};cost+=v.fixed_cost
        for c in customers:
            q=problem.customers[c];distance=hypot(xy[0]-q.x,xy[1]-q.y)
            if arc_rounding_scale:distance=round(distance*arc_rounding_scale)/arc_rounding_scale
            t=max(t+distance,q.tw_start);starts[c]=t
            if t>q.tw_end+1e-9:errors.append(f'route:{index}:time_window:{c}')
            saved=r.get('service_starts',{}).get(str(c),r.get('service_starts',{}).get(c))
            if saved is not None and not isclose(float(saved),t,abs_tol=1e-7,rel_tol=1e-10):errors.append(f'route:{index}:service_start:{c}')
            e=distance*(v.energy_alpha+v.energy_beta*load)
            cost+=distance;energy+=e;co2+=e*v.emission_factor*problem.zone_factors.get(q.arc_zone,1)*problem.shift_factors[s.idx]
            t+=q.service;load-=q.demand;xy=(q.x,q.y)
        distance=hypot(xy[0]-problem.depot_xy[0],xy[1]-problem.depot_xy[1])
        if arc_rounding_scale:distance=round(distance*arc_rounding_scale)/arc_rounding_scale
        t+=distance
        e=distance*(v.energy_alpha+v.energy_beta*max(0,load));cost+=distance;energy+=e
        co2+=e*v.emission_factor*problem.zone_factors.get(problem.customers[customers[-1]].arc_zone,1)*problem.shift_factors[s.idx]
        overtime=max(0.,t-s.end);ratio=overtime/(problem.eta*s.length+problem.epsilon)
        ot_sum+=overtime;ot_ratio+=ratio
        if overtime>problem.eta*s.length+1e-9:errors.append(f'route:{index}:depot_return')
        route_info.append({'route':index,'return_time':t,'shift_end':s.end,'slack':s.end-t,'overtime':overtime,'service_starts':starts})
    counts=Counter(seen)
    for c in problem.customers:
        if counts[c]!=1:errors.append(f'coverage:{c}:{counts[c]}')
    for c in counts:
        if c not in problem.customers:errors.append(f'unknown_customer:{c}')
    values={'cost':cost,'energy':energy,'co2':co2,'overtime_sum':ot_sum,'overtime_ratio_sum':ot_ratio}
    if references is not None:
        if any(not isfinite(float(x)) or x<=0 for x in references):errors.append('invalid_references')
        else:values['score']=sum(w*x/ref for w,x,ref in zip(problem.objective_weights,[cost,energy,co2],references))/sum(problem.objective_weights)+problem.overtime_penalty*ot_ratio
    if reported:
        for key,value in values.items():
            if key in reported and not isclose(value,float(reported[key]),abs_tol=1e-6,rel_tol=1e-8):errors.append('reported_mismatch:'+key)
    return {'valid':not errors,'errors':errors,'recomputed':values,'routes':route_info,'strict_duty':not errors and ot_sum<=1e-9}


def serialize_solution(sol):
    return [{'shift_id':r.shift_id,'vehicle_type':r.vehicle_type,'customers':r.customers,
             'service_starts':r.metrics.service_times,'solver_return_time':r.metrics.return_time} for r in sol.routes]
