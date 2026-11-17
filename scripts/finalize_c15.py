"""C15 core evidence and portal validation are explicitly separate commands."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
COMMANDS=[
 ['validate_master_runs.py','generated_submission/master_runs.csv','--manifest','configs/experiment_manifest_submission_primary.csv','--runtime-tolerance-s','10','--allow-locked-runtime-exception'],
 ['aggregate_results.py','generated_submission/master_runs.csv','generated_submission/summary_by_method.csv'],
 ['run_stats.py','generated_submission/master_runs.csv','generated_submission/master_stats.csv'],
 ['build_latex_tables.py','generated_submission/summary_by_method.csv','generated_submission/master_stats.csv','--output-dir','generated_submission'],
 ['build_robustness_evidence.py','generated_submission/master_runs.csv','--output-dir','generated_submission'],
 ['build_claim_macros.py','generated_submission/summary_by_method.csv','--output','generated_submission/claim_macros.tex'],
 ['build_claim_evidence_map.py','--summary','generated_submission/summary_by_method.csv','--stats','generated_submission/master_stats.csv','--output','generated_submission/claim_evidence_map.csv'],
 ['build_operational_shift_evidence.py','--ledger','configs/operational_shift_stress_locked.csv','--solomon-audit','configs/solomon_equal_3shift_feasibility_audit.csv','--out','generated_submission'],
 ['build_parameter_sensitivity_evidence.py','--ledger','configs/parameter_sensitivity_locked.csv','--out','generated_submission'],
 ['build_anytime_profile_evidence.py','--ledger','configs/anytime_profile_locked.csv','--out','generated_submission'],
 ['analyze_c15_primary.py'],
]
def core():
    env=os.environ.copy();env['SWEVO_GENERATED_DIR']='generated_submission';env['PYTHONPATH']=str(ROOT/'src');env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1'
    for args in COMMANDS:
        subprocess.run([sys.executable,str(ROOT/'scripts'/args[0]),*args[1:]],cwd=ROOT,env=env,check=True)
    # Core provenance excludes historical diagnostics and all portal assets.
    inputs=[ROOT/'generated_submission/master_runs.csv',*(ROOT/'configs').glob('*'),*(ROOT/'src').rglob('*.py'),*(ROOT/'scripts').glob('*.py')]
    outputs=[p for p in (ROOT/'generated_submission').glob('*') if p.is_file() and (p.suffix in ['.csv','.tex']) and 'historical' not in p.name and ' (1)' not in p.name]
    record=lambda p:dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
    payload=dict(schema='c15-core-provenance-v1',inputs=[record(p) for p in sorted(inputs) if p.is_file()],outputs=[record(p) for p in sorted(outputs)])
    (ROOT/'generated_submission/c15_core_provenance.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('C15_CORE=PASS; no archive or portal input required')
def portal():
    from pypdf import PdfReader
    m=ROOT/'manuscript'
    for name in ['manuscript.pdf','elsarticle-template-num.tex','cas-refs.bib','elsarticle-num.bst','Highlights.txt','COVER_LETTER.txt','graphical_abstract_source.tex','graphical_abstract.pdf','graphical_abstract.png','c15_build_provenance.json']:
        assert (m/name).is_file() and (m/name).stat().st_size>0,name
    bullets=[s.lstrip('- ').strip() for s in (m/'Highlights.txt').read_text(encoding='utf-8').splitlines() if s.startswith('- ')]
    assert len(bullets)==5 and max(map(len,bullets))<=85
    tex=(m/'elsarticle-template-num.tex').read_text(encoding='utf-8')
    assert '108 independent' not in tex and '36 independent' not in tex
    pdf=PdfReader(m/'manuscript.pdf');text='\n'.join(p.extract_text() or '' for p in pdf.pages)
    assert '??' not in text and 'Missing file:' not in text
    provenance=json.loads((m/'c15_build_provenance.json').read_text())
    for rel,value in provenance['files'].items(): assert sha(ROOT/rel)==value,rel
    assert 'References' in text and len(pdf.pages)>10
    print(f'C15_PORTAL=PASS pages={len(pdf.pages)}')
def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--core',action='store_true');group.add_argument('--portal',action='store_true');group.add_argument('--historical',type=Path,help='Optional historical full ledger, never required by core');a=ap.parse_args()
    if a.core:core()
    elif a.portal:portal()
    else:subprocess.run([sys.executable,str(ROOT/'scripts/build_submission_strengthening_evidence.py'),'--primary',str(ROOT/'generated_submission/master_runs.csv'),'--full',str(a.historical),'--out',str(ROOT/'optional_historical')],check=True)
if __name__=='__main__':main()
