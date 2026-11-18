"""Serial resumable job graph, no parallel optimizer processes."""
from pathlib import Path
import os,sys,subprocess,json,time
from run_ablation import atomic,utc
ROOT=Path(__file__).resolve().parents[1]


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--start-delay-s',type=float,default=0);a=ap.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONUTF8']='1';os.environ.pop('PYTHONPATH',None)
    f=ROOT.parent/'frozen/EDE_SUSTAINABLE_VRPTW_C16.0';s=ROOT/'scripts'
    tasks=[
        ('frozen_ablation',[str(s/'run_ablation.py'),'--project-root',str(f/'workspace'),'--campaign',str(f/'ablation'),'--output',str(ROOT/'results/ablation_execution')]),
        ('frozen_original_analysis',[str(s/'analyze_ablation.py'),'--campaign',str(f/'ablation'),'--execution',str(ROOT/'results/ablation_execution'),'--project-root',str(f/'workspace'),'--out',str(ROOT/'results/ablation_final')]),
        ('frozen_nine_instance_analysis',[str(s/'analyze_c17_ablation.py')]),
        ('boundary_selection',[str(s/'run_c17_cohort.py'),'--manifest',str(ROOT/'protocol/boundary_only_540.csv'),'--selection-from-manifest','--output',str(ROOT/'results/boundary_selection')]),
        ('external_narrow',[str(s/'external_hgs.py'),'--manifest',str(ROOT/'protocol/external_narrow_180.csv'),'--output',str(ROOT/'results/external_narrow')]),
        ('new_cohort_analysis',[str(s/'analyze_new_cohorts.py')])]
    job_id='C17_SERIAL_'+utc().replace(':','').replace('.','_');state={'job_id':job_id,'pid':os.getpid(),'started_utc':utc(),'status':'running','tasks':[name for name,_ in tasks]}
    atomic(ROOT/'control/SUPERVISOR.json',state)
    if a.start_delay_s:
        state.update(status='waiting_for_delivery',delay_s=a.start_delay_s);atomic(ROOT/'control/SUPERVISOR.json',state)
        time.sleep(a.start_delay_s)
        state.update(status='running');atomic(ROOT/'control/SUPERVISOR.json',state)
    for name,args in tasks:
        state.update(current_task=name,task_started_utc=utc());atomic(ROOT/'control/SUPERVISOR.json',state)
        print(utc(),name,'START',flush=True)
        child=subprocess.Popen([sys.executable,*args],cwd=ROOT)
        state['child_pid']=child.pid;atomic(ROOT/'control/SUPERVISOR.json',state)
        code=child.wait()
        state.update(last_task=name,last_exit_code=code,last_task_ended_utc=utc());atomic(ROOT/'control/SUPERVISOR.json',state)
        subprocess.run([sys.executable,str(s/'c17_monitor.py')],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
        if code:
            state.update(status='failed',failure_task=name);atomic(ROOT/'control/SUPERVISOR.json',state);raise SystemExit(code)
    state.update(status='completed_queued_tasks',ended_utc=utc(),note='Remaining operational/full-objective/author gates still prevent submission promotion')
    atomic(ROOT/'control/SUPERVISOR.json',state)
if __name__=='__main__':main()
