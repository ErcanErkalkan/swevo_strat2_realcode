#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def validate_locked(ledger: pd.DataFrame, audit: pd.DataFrame) -> dict[str, object]:
    required = {'run_id','tier','customer_count','structure','scenario_id','seed','window_adjustments','window_shrinkages','accepted','strict_duty','v_cap','v_tw','v_shift','overtime_sum','shift1_start','shift1_end','shift2_start','shift2_end','shift3_start','shift3_end'}
    missing = required - set(ledger.columns)
    if missing: raise SystemExit(f'operational ledger missing columns: {sorted(missing)}')
    if len(ledger) != 18: raise SystemExit(f'expected 18 locked operational rows, found {len(ledger)}')
    expected_grid={(t,s,se) for t in ['small','medium','large'] for s in ['clustered','mixed','random'] for se in [1,2]}
    actual={(str(r.tier),str(r.structure),int(r.seed)) for r in ledger.itertuples(index=False)}
    if actual != expected_grid: raise SystemExit(f'operational grid mismatch missing={sorted(expected_grid-actual)} extra={sorted(actual-expected_grid)}')
    if set(ledger['scenario_id'].astype(str)) != {'S3_mixed_fleet_arc'}: raise SystemExit('operational ledger must use S3_mixed_fleet_arc only')
    if not (pd.to_numeric(ledger['window_shrinkages']) == 0).all(): raise SystemExit('window shrinkage detected in locked operational ledger')
    for a,b in [('shift1_end','shift2_start'),('shift2_end','shift3_start')]:
        if not (pd.to_numeric(ledger[a]) == pd.to_numeric(ledger[b])).all(): raise SystemExit(f'noncontiguous shifts: {a}!={b}')
    if not (pd.to_numeric(ledger['shift1_start']) == 0).all(): raise SystemExit('shift1_start must be zero')
    if not (pd.to_numeric(ledger['shift1_end']) < pd.to_numeric(ledger['shift2_end'])).all(): raise SystemExit('invalid shift ordering')
    if not (pd.to_numeric(ledger['shift2_end']) < pd.to_numeric(ledger['shift3_end'])).all(): raise SystemExit('invalid shift ordering')
    if not (pd.to_numeric(ledger['accepted']) == 1).all(): raise SystemExit('not all operational runs are hard-feasible')
    for c in ['v_cap','v_tw','v_shift']:
        if not (pd.to_numeric(ledger[c]).abs() <= 1e-9).all(): raise SystemExit(f'nonzero hard violation in {c}')
    if len(audit)!=3 or set(audit['instance'].astype(str)) != {'C101','R101','RC101'}: raise SystemExit('Solomon equal-3-shift audit must contain C101/R101/RC101')
    if not (pd.to_numeric(audit['single_customer_infeasible']) > 0).all(): raise SystemExit('Solomon partition audit must demonstrate infeasible customers in every audited instance')
    strict=int(pd.to_numeric(ledger['strict_duty']).sum())
    return {
        'runs': int(len(ledger)), 'hard_feasible_runs': int(pd.to_numeric(ledger['accepted']).sum()),
        'strict_duty_runs': strict, 'bounded_overtime_runs': int(len(ledger)-strict),
        'max_overtime_sum': float(pd.to_numeric(ledger['overtime_sum']).max()),
        'window_shrinkage_runs': int((pd.to_numeric(ledger['window_shrinkages'])>0).sum()),
        'tiers': 3, 'structures': 3, 'paired_seeds': 2,
        'solomon_audit_infeasible_counts': {str(r.instance): int(r.single_customer_infeasible) for r in audit.itertuples(index=False)},
    }

def build_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for tier in ['small','medium','large']:
        g=ledger[ledger['tier'].astype(str)==tier]
        rows.append({'scope':tier,'runs':len(g),'hard_feasible_runs':int(g['accepted'].sum()),'strict_duty_runs':int(g['strict_duty'].sum()),'bounded_overtime_runs':int(len(g)-g['strict_duty'].sum()),'max_overtime_sum':float(g['overtime_sum'].max()),'max_v_cap':float(g['v_cap'].max()),'max_v_tw':float(g['v_tw'].max()),'max_v_shift':float(g['v_shift'].max())})
    g=ledger
    rows.append({'scope':'all','runs':len(g),'hard_feasible_runs':int(g['accepted'].sum()),'strict_duty_runs':int(g['strict_duty'].sum()),'bounded_overtime_runs':int(len(g)-g['strict_duty'].sum()),'max_overtime_sum':float(g['overtime_sum'].max()),'max_v_cap':float(g['v_cap'].max()),'max_v_tw':float(g['v_tw'].max()),'max_v_shift':float(g['v_shift'].max())})
    return pd.DataFrame(rows)

def table_tex(summary: pd.DataFrame) -> str:
    labels={"small":"Small (100)","medium":"Medium (200)","large":"Large (400)","all":"All"}
    bs=chr(92)
    lines=[
        f"{bs}begin{{table}}[H]",
        f"{bs}centering",
        f"{bs}small",
        f"{bs}setlength{{{bs}tabcolsep}}{{5pt}}",
        f"{bs}renewcommand{{{bs}arraystretch}}{{1.12}}",
        f"{bs}begin{{tabular}}{{lrrrrr}}",
        f"{bs}toprule",
        f"{bs}textbf{{Scope}} & {bs}textbf{{Runs}} & {bs}textbf{{Hard feasible}} & {bs}textbf{{Zero overtime}} & {bs}textbf{{Bounded OT}} & {bs}textbf{{Max OT}} {bs}{bs}",
        f"{bs}midrule",
    ]
    for r in summary.itertuples(index=False):
        lines.append(f"{labels[str(r.scope)]} & {int(r.runs)} & {int(r.hard_feasible_runs)} & {int(r.strict_duty_runs)} & {int(r.bounded_overtime_runs)} & {float(r.max_overtime_sum):.3f} {bs}{bs}")
    lines += [
        f"{bs}bottomrule",
        f"{bs}end{{tabular}}",
        f"{bs}caption{{Secondary operational-semantics stress validation with three genuine non-overlapping shifts. All runs satisfy the hard feasibility predicate; bounded overtime is permitted by the model and is reported separately from strict zero-overtime duty compliance.}}",
        f"{bs}label{{tab:operational_shift_stress}}",
        f"{bs}end{{table}}",
    ]
    return "\n".join(lines)+"\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ledger',default='data/secondary/operational_shift_stress_locked.csv')
    ap.add_argument('--solomon-audit',default='data/secondary/solomon_equal_3shift_feasibility_audit.csv')
    ap.add_argument('--out',default='results/primary')
    args=ap.parse_args()
    ledger_path=(ROOT/args.ledger).resolve() if not Path(args.ledger).is_absolute() else Path(args.ledger)
    audit_path=(ROOT/args.solomon_audit).resolve() if not Path(args.solomon_audit).is_absolute() else Path(args.solomon_audit)
    out=(ROOT/args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ledger=pd.read_csv(ledger_path); audit=pd.read_csv(audit_path)
    checks=validate_locked(ledger,audit); summary=build_summary(ledger)
    summary_path=out/'operational_shift_stress_summary.csv'; summary.to_csv(summary_path,index=False)
    table_path=out/'table_operational_shift_stress.tex'; table_path.write_text(table_tex(summary),encoding='utf-8')
    prov={'schema':'ede-operational-shift-evidence-v1','scope':'secondary_operational_semantics','claim_boundary':'implementation-level feasibility only; not real-world deployment validation or optimizer-superiority evidence','inputs':[{'path':ledger_path.relative_to(ROOT).as_posix(),'sha256':sha256(ledger_path),'bytes':ledger_path.stat().st_size},{'path':audit_path.relative_to(ROOT).as_posix(),'sha256':sha256(audit_path),'bytes':audit_path.stat().st_size}], 'checks':checks,'outputs':[{'path':summary_path.relative_to(ROOT).as_posix(),'sha256':sha256(summary_path),'bytes':summary_path.stat().st_size},{'path':table_path.relative_to(ROOT).as_posix(),'sha256':sha256(table_path),'bytes':table_path.stat().st_size}]}
    prov_path=out/'operational_shift_evidence_provenance.json'; prov_path.write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(f"operational_shift_evidence=PASS runs={checks['runs']} hard_feasible={checks['hard_feasible_runs']} strict={checks['strict_duty_runs']} max_ot={checks['max_overtime_sum']:.6f}")
if __name__=='__main__': main()
