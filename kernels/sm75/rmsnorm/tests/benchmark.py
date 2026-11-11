"""Small standalone operator timings; never imply full-model speedup."""
import argparse, json, statistics, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sm75_norm import rmsnorm, reference

def gpu_us(fn,repeats):
    for _ in range(3): fn()
    values=[]
    for _ in range(5):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for __ in range(repeats): fn()
        end.record(); end.synchronize()
        values.append(start.elapsed_time(end)*1000/repeats)
    return {'median_us':statistics.median(values),'min_us':min(values),'max_us':max(values)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='benchmark.json'); p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--rows',type=int,default=8); a=p.parse_args()
    if not 1<=a.repeats<=50 or not 1<=a.rows<=128: raise ValueError('bounded benchmark limits')
    torch.set_num_threads(1); torch.manual_seed(711)
    results=[]
    for d in (8192,32768):
        for relu2 in (False,True):
            x=torch.randn(a.rows,d,device='cuda',requires_grad=True)
            g=torch.randn(d,device='cuda',requires_grad=True); dy=torch.randn_like(x)
            for training in (False,True):
                def candidate():
                    y=rmsnorm(x,g,relu2=relu2)
                    return torch.autograd.grad(y,(x,g),dy) if training else y
                def baseline():
                    y=reference(x,g,relu2=relu2)
                    return torch.autograd.grad(y,(x,g),dy) if training else y
                torch.testing.assert_close(rmsnorm(x,g,relu2=relu2),reference(x,g,relu2=relu2),rtol=2e-5,atol=2e-6)
                b=gpu_us(baseline,a.repeats); c=gpu_us(candidate,a.repeats)
                results.append({'rows':a.rows,'width':d,'relu2':relu2,'forward_and_backward':training,
                    'baseline':b,'sm75':c,'ratio_baseline_over_sm75':b['median_us']/c['median_us']})
    report={'gpu':torch.cuda.get_device_name(),'scope':'synthetic isolated operator, not full-model',
            'warning':'Timing is invalid as a clean speed estimate if other GPU work overlapped.',
            'results':results,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
