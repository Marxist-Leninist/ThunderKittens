"""Bounded correctness checks. No checkpoint/model loads or package installs."""
import argparse
import gc
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sm75_norm import rmsnorm, reference

def metrics(got,want):
    err=(got.double()-want.double()).abs()
    denom=want.double().square().mean().sqrt().clamp_min(1e-30)
    return {'max_abs':err.max().item() if err.numel() else 0.0,
            'normalized_rms':(err.square().mean().sqrt()/denom).item() if err.numel() else 0.0}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='correctness.json')
    args=ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(711)
    report={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'capability':torch.cuda.get_device_capability(),'tests':[]}
    shapes=[(1,1),(3,7),(2,31),(3,33),(4,257),(65,257),(1,8192),(8,8192),(1,32768),(8,32768)]
    for relu2 in (False,True):
        for rows,d in shapes:
            xc=torch.randn(rows,d)
            gcpu=torch.randn(d)
            dyc=torch.randn(rows,d)
            x=xc.cuda().requires_grad_(); g=gcpu.cuda().requires_grad_(); dy=dyc.cuda()
            y=rmsnorm(x,g,relu2=relu2); dx,dg=torch.autograd.grad(y,(x,g),dy)
            yr=reference(x,g,relu2=relu2); dxr,dgr=torch.autograd.grad(yr,(x,g),dy)
            # Independent float64 mathematical reference on CPU.
            xd=xc.double().requires_grad_(); gd=gcpu.double().requires_grad_()
            h=xd.relu().square() if relu2 else xd
            yd=h*torch.rsqrt(h.square().mean(-1,keepdim=True)+1e-6)*gd
            dxd,dgd=torch.autograd.grad(yd,(xd,gd),dyc.double())
            outputs={}
            for name,a,b,c,rtol,atol in [('forward',y,yr,yd,2e-5,2e-6),
                 ('dx',dx,dxr,dxd,1e-4,1e-5),('dgamma',dg,dgr,dgd,2e-4,2e-5)]:
                torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
                ac=a.detach().cpu()
                torch.testing.assert_close(ac.double(),c.detach(),rtol=rtol,atol=atol)
                outputs[name]={'vs_fp32':metrics(ac,b.detach().cpu()),'vs_fp64':metrics(ac,c.detach())}
            report['tests'].append({'shape':[rows,d],'relu2':relu2,'result':'pass','errors':outputs})
            del x,g,dy,y,dx,dg,yr,dxr,dgr,xd,gd,h,yd,dxd,dgd
    for relu2 in (False,True):
        for kind,scale in [('zeros',0.0),('tiny',1e-5),('large',100.0),('constant',1.0),('zero_gamma',1.0)]:
            x=(torch.ones(3,257,device='cuda') if kind=='constant' else torch.randn(3,257,device='cuda')*scale).requires_grad_()
            g=(torch.zeros(257,device='cuda') if kind=='zero_gamma' else torch.randn(257,device='cuda')).requires_grad_()
            dy=torch.randn_like(x)
            a=rmsnorm(x,g,relu2=relu2); da=torch.autograd.grad(a,(x,g),dy)
            b=reference(x,g,relu2=relu2); db=torch.autograd.grad(b,(x,g),dy)
            for actual,want in zip((a,*da),(b,*db)):
                torch.testing.assert_close(actual,want,rtol=2e-4,atol=2e-5)
            report['tests'].append({'edge_case':kind,'relu2':relu2,'result':'pass'})
    # Non-default stream and explicit stream ownership.
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x=torch.randn(2,8192,device='cuda',requires_grad=True)
        g=torch.randn(8192,device='cuda',requires_grad=True)
        y=rmsnorm(x,g); z=reference(x,g)
        dy=torch.ones_like(x)
        dx,dg=torch.autograd.grad(y,(x,g),dy)
        dxr,dgr=torch.autograd.grad(z,(x,g),dy)
    stream.synchronize()
    for a,b in ((y,z),(dx,dxr),(dg,dgr)):
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
    report['tests'].append({'case':'non_default_stream','result':'pass'})
    for bad in (x.t(),x.half()):
        try: rmsnorm(bad,torch.ones(bad.shape[-1],device='cuda'))
        except (ValueError,TypeError): pass
        else: raise AssertionError('invalid input accepted')
    report['tests'].append({'case':'layout_and_dtype_rejection','result':'pass'})
    # Preserve NaNs so fusion cannot conceal an upstream numerical failure.
    for relu2 in (False,True):
        nx=torch.randn(2,33,device='cuda',requires_grad=True)
        with torch.no_grad(): nx[0,5]=float('nan')
        ng=torch.ones(33,device='cuda',requires_grad=True)
        na=rmsnorm(nx,ng,relu2=relu2); nb=reference(nx,ng,relu2=relu2)
        nad=torch.autograd.grad(na,(nx,ng),torch.ones_like(nx))
        nbd=torch.autograd.grad(nb,(nx,ng),torch.ones_like(nx))
        for aa,bb in zip((na,*nad),(nb,*nbd)):
            torch.testing.assert_close(aa,bb,rtol=1e-4,atol=1e-5,equal_nan=True)
        report['tests'].append({'case':'nan_propagation','relu2':relu2,'result':'pass'})
    for shape in ((257,),(2,3,257)):
        rx=torch.randn(shape,device='cuda',requires_grad=True)
        rg=torch.randn(257,device='cuda',requires_grad=True)
        ra=rmsnorm(rx,rg,eps=3e-4); rb=reference(rx,rg,eps=3e-4)
        rd=torch.randn((shape[-1],*shape[:-1]),device='cuda').movedim(0,-1)
        rad=torch.autograd.grad(ra,(rx,rg),rd); rbd=torch.autograd.grad(rb,(rx,rg),rd)
        for aa,bb in zip((ra,*rad),(rb,*rbd)):
            torch.testing.assert_close(aa,bb,rtol=1e-4,atol=1e-5)
        report['tests'].append({'case':'rank_eps_gradient_layout','shape':list(shape),'result':'pass'})
    neg=-torch.rand(2,33,device='cuda',requires_grad=True)
    ng=torch.ones(33,device='cuda',requires_grad=True)
    no=rmsnorm(neg,ng,relu2=True)
    nd=torch.autograd.grad(no.sum(),(neg,ng))
    assert all(torch.count_nonzero(v).item()==0 for v in (no,*nd))
    report['tests'].append({'case':'negative_only_fused','result':'pass'})
    # Empty leading dimension remains differentiable.
    e=torch.empty(0,257,device='cuda',requires_grad=True)
    eg=torch.ones(257,device='cuda',requires_grad=True)
    eo=rmsnorm(e,eg); edx,edg=torch.autograd.grad(eo.sum(),(e,eg))
    assert edx.shape == e.shape and torch.count_nonzero(edg).item()==0
    report['tests'].append({'case':'empty_rows','result':'pass'})
    c=torch.randn(2,7,requires_grad=True); cg=torch.ones(7,requires_grad=True)
    torch.testing.assert_close(rmsnorm(c,cg,fallback=True),reference(c,cg))
    cd=torch.autograd.grad(rmsnorm(c,cg,fallback=True).sum(),(c,cg))
    cr=torch.autograd.grad(reference(c,cg).sum(),(c,cg))
    for aa,bb in zip(cd,cr): torch.testing.assert_close(aa,bb)
    report['tests'].append({'case':'explicit_cpu_fallback','result':'pass'})
    torch.cuda.synchronize()
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    report['passed']=len(report['tests'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tests'},indent=2))

if __name__=='__main__': main()
