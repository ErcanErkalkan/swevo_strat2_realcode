"""Read-only status with audit-generated coverage; never infer liveness from a log."""
from pathlib import Path
import json,os,sys,ctypes,hashlib
from run_ablation import atomic,utc
from audit_ablation import audit
ROOT=Path(__file__).resolve().parents[1]


def live(pid):
    if not pid:return False
    if os.name=='nt':
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        code=ctypes.c_ulong();ok=kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code));kernel.CloseHandle(ctypes.c_void_p(handle));return bool(ok and code.value==259)
    try:os.kill(int(pid),0);return True
    except OSError:return False


def refresh(root=ROOT):
    output=root/'results/ablation_execution'
    campaign=root.parent/'frozen/EDE_SUSTAINABLE_VRPTW_C16.0/ablation'
    if not campaign.exists():campaign=root.parent/'baseline/EDE_SUSTAINABLE_VRPTW_C16.0/ablation'
    audit_status=audit(campaign,root/'results/ablation',output)
    status=json.loads((root/'STATUS.json').read_text());status.update(updated_utc=utc(),state='PARTIAL_RESEARCH_CANDIDATE',complete=False,ablation=audit_status)
    job_path=root/'control/JOB.json';job=json.loads(job_path.read_text()) if job_path.exists() else {}
    active=[];terminal=[]
    for p in output.glob('*.json'):
        rec=json.loads(p.read_text());(active if rec['result_status']=='started' else terminal).append(rec['run_id'])
    supervisor_path=root/'control/SUPERVISOR.json'
    supervisor=json.loads(supervisor_path.read_text()) if supervisor_path.exists() else {}
    status['job']={**job,'supervisor':supervisor,'alive_at_last_check':live(job.get('pid')),'checked_utc':utc(),'active_run_ids':active,'terminal_run_ids':terminal}
    for cohort,manifest,folder in [('boundary_selection','boundary_only_540.csv','boundary_selection'),('external_narrow','external_narrow_180.csv','external_narrow')]:
        import csv
        with (root/'protocol'/manifest).open() as f:design=list(csv.DictReader(f))
        n=len(design)
        files=list((root/'results'/folder).glob('*.json'));counts={s:0 for s in ['completed','failed','started']}
        for p in files:
            s=json.loads(p.read_text()).get('status');counts[s]=counts.get(s,0)+1
        status['new_cohorts'][cohort]={'planned':n,**counts,'missing':n-counts['completed'],'total_nominal_hours':sum(float(r['walltime_cap_s']) for r in design)/3600}
    atomic(root/'STATUS.json',status);print(json.dumps(status,indent=2));return status
if __name__=='__main__':refresh()
