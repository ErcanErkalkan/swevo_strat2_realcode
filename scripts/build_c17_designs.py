from pathlib import Path
import sys,json,hashlib,copy
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swevo_suite.benchmark import build_problem,load_scenarios
from swevo_suite.manifest import load_manifest
from swevo_suite.models import Shift,Route
from swevo_suite.c17_verifier import verify
from dataclasses import replace
from math import hypot


def main():
    base=pd.read_csv(ROOT/'data/ablation/manifest.csv');base=base[base.method_id=='EDE'].drop_duplicates(['instance_id','scenario_id']).copy()
    rows=[]
    for r in base.to_dict('records'):
        for seed in range(11,21):
            for selection in (['boundary','random'] if seed%2 else ['random','boundary']):
                q=dict(r,seed=seed,phase='c17_boundary_'+selection,status='planned',selection=selection)
                rows.append(q)
    pd.DataFrame(rows).to_csv(ROOT/'protocol/boundary_only_540.csv',index=False)
    ext=[]
    for r in base[base.scenario_id=='S1_balanced'].to_dict('records'):
        for seed in range(11,21):
            for method in (['EDE_NARROW','PYVRP_HGS'] if seed%2 else ['PYVRP_HGS','EDE_NARROW']):ext.append(dict(r,seed=seed,method_id=method,phase='c17_external_narrow',status='planned'))
    pd.DataFrame(ext).to_csv(ROOT/'protocol/external_narrow_180.csv',index=False)
    # A unique manifest per selection is essential: RunPlan IDs omit phase.
    for selection in ['boundary','random']:
        pd.DataFrame([r for r in rows if r['selection']==selection]).to_csv(ROOT/f'protocol/boundary_{selection}_270.csv',index=False)
    structure=[];witnesses=[]
    for plan in load_manifest(ROOT/'data/ablation/manifest.csv'):
        if plan.method_id!='EDE' or plan.seed!=1:continue
        original=build_problem(plan,load_scenarios(),require_real=True);h=original.shifts[0].end
        for regime in ['overlapping','sequential','binding_witness']:
            p=copy.deepcopy(original)
            if regime=='sequential':p.shifts=[Shift(i+1,i*h/3,(i+1)*h/3) for i in range(3)]
            routes=[];missing=[]
            for cid,c in p.customers.items():
                choices=[]
                for s in p.shifts:
                    for name,v in p.vehicle_types.items():
                        arrival=max(s.start+hypot(c.x-p.depot_xy[0],c.y-p.depot_xy[1]),c.tw_start)
                        ret=arrival+c.service+hypot(c.x-p.depot_xy[0],c.y-p.depot_xy[1])
                        if arrival<=c.tw_end+1e-9 and c.demand<=v.capacity and ret<=s.end+1e-9:choices.append((ret,s.idx,name))
                if not choices:missing.append(cid)
                else:
                    ret,sid,name=min(choices);routes.append({'shift_id':sid,'vehicle_type':name,'customers':[cid]})
            # Binding witness preserves every native customer window. Tighten
            # used shift ends only to the largest singleton return, never below
            # that constructive hard-feasible witness. Not a solver outcome.
            if regime=='binding_witness' and not missing:
                initial=verify(p,routes)
                ends={s.idx:max([r['return_time'] for r,q in zip(initial['routes'],routes) if q['shift_id']==s.idx] or [s.end]) for s in p.shifts}
                p.shifts=[Shift(s.idx,s.start,ends[s.idx]) for s in p.shifts]
            audit=verify(p,routes)
            structure.append({'instance_id':plan.instance_id,'scenario_id':plan.scenario_id,'regime':regime,'source':'synthetic transformation of public benchmark','customers':len(p.customers),'singleton_uncovered':len(missing),'constructive_feasible':audit['valid'],'binding_witness_routes':sum(abs(x['slack'])<1e-7 for x in audit['routes']),'minimum_witness_return_slack':min([x['slack'] for x in audit['routes']] or [float('nan')]),'interpretation':'structural witness; NOT optimization/operational effectiveness'})
            witnesses.append({'instance_id':plan.instance_id,'scenario_id':plan.scenario_id,'regime':regime,'shifts':[{'idx':s.idx,'start':s.start,'end':s.end} for s in p.shifts],'routes':routes,'uncovered_customer_ids':missing,'audit':audit})
    pd.DataFrame(structure).to_csv(ROOT/'results/c17/operational_structural_audit.csv',index=False)
    (ROOT/'results/c17/operational_witnesses.json').write_text(json.dumps(witnesses,indent=2)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'protocol').glob('*.csv')}
    (ROOT/'protocol/MANIFEST_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('boundary=540, external_narrow=180; structural cases=',len(structure))
if __name__=='__main__':main()
