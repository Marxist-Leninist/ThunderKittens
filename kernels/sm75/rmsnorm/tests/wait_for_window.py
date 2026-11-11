"""Bounded read-only readiness polling, followed by this task's tiny checks only."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seconds',type=int,default=180)
    p.add_argument('--benchmark',action='store_true'); a=p.parse_args()
    if not 0<a.seconds<=600: raise ValueError('polling is bounded at 600 seconds')
    root=Path(__file__).resolve().parents[1]
    deadline=time.monotonic()+a.seconds; ready=0; samples=[]
    while time.monotonic()<deadline:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True)
        util,free=map(int,out.strip().splitlines()[0].split(','))
        row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'utilization':util,'free_mib':free}
        samples.append(row); print(json.dumps(row),flush=True)
        ready=ready+1 if util<=15 and free>=1024 else 0
        if ready>=2:
            env=os.environ.copy(); env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            cmd=[sys.executable,str(root/'tests'/'check_correctness.py'),'--output',str(root/'correctness.json')]
            rc=subprocess.run(cmd,cwd=root,env=env,timeout=90).returncode
            if rc: return rc
            if a.benchmark:
                util2,free2=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0].split(','))
                if util2<=15 and free2>=1024:
                    rc=subprocess.run([sys.executable,str(root/'tests'/'benchmark.py'),'--output',str(root/'benchmark.json'),'--repeats','5'],cwd=root,env=env,timeout=90).returncode
                else: print('BENCHMARK_SKIPPED_WINDOW_CLOSED',flush=True)
            (root/'readiness.json').write_text(json.dumps({'samples':samples,'result':'ran_tiny_correctness','exit_code':rc},indent=2)+'\n')
            return rc
        time.sleep(5)
    (root/'readiness.json').write_text(json.dumps({'samples':samples,'result':'skipped_no_safe_window'},indent=2)+'\n')
    print('SKIPPED_NO_SAFE_WINDOW',flush=True)
    return 0
if __name__=='__main__': sys.exit(main())
