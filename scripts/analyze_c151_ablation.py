"""Gate the unchanged preregistered final analyzer behind scientific validation."""
from pathlib import Path
import argparse,subprocess,sys,json,importlib.util
from audit_c15_ablation import audit,sha
def require_final(campaign,out,execution=None):
    status=audit(campaign,out,execution)
    if not status['scientific_closure_ready']:raise ValueError('Final ablation inference blocked: require complete validated records and full matched runtime handling')
    return status
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--execution',type=Path);ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    require_final(a.campaign,a.out/'validation',a.execution)
    c=a.campaign
    lock=json.loads((c/'confirmatory_ablation_source_lock.json').read_text())
    if sha(c/'confirmatory_ablation_manifest_1080_cli.csv')!=lock['manifest_sha256'] or sha(c/'representative_instance_selection.csv')!=lock['selection_sha256']:raise ValueError('Locked manifest/selection hash mismatch')
    spec=importlib.util.spec_from_file_location('original_locked_runner',c/'run_confirmatory_ablation.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_source_lock(a.project_root,c/'confirmatory_ablation_source_lock.json')
    module.verify_benchmarks(a.project_root,c/'representative_instance_selection.csv')
    subprocess.run([sys.executable,str(c/'analyze_confirmatory_ablation.py'),'--project-root',str(a.project_root),'--manifest',str(c/'confirmatory_ablation_manifest_1080_cli.csv'),'--results',str(a.out/'validation/c15_ablation_validated_records.csv'),'--selection',str(c/'representative_instance_selection.csv'),'--source-lock',str(c/'confirmatory_ablation_source_lock.json'),'--out',str(a.out/'final_analysis')],check=True)
if __name__=='__main__':main()
