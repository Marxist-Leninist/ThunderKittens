"""CPU-only wrapper and analytic-contract tests, not GPU kernel validation."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sm75_norm import rmsnorm, reference
torch.set_num_threads(1)

class ContractTests(unittest.TestCase):
    def test_cpu_is_rejected_by_native_path(self):
        with self.assertRaises(ValueError): rmsnorm(torch.ones(2,7),torch.ones(7))
    def test_fallback_forward(self):
        x=torch.randn(2,3,7); g=torch.randn(7)
        for fused in (False,True):
            torch.testing.assert_close(rmsnorm(x,g,relu2=fused,fallback=True),reference(x,g,relu2=fused))
    def test_fallback_backward(self):
        x=torch.randn(2,33,requires_grad=True); g=torch.randn(33,requires_grad=True)
        for fused in (False,True):
            a=torch.autograd.grad(rmsnorm(x,g,relu2=fused,fallback=True).sum(),(x,g))
            b=torch.autograd.grad(reference(x,g,relu2=fused).sum(),(x,g))
            for aa,bb in zip(a,b): torch.testing.assert_close(aa,bb)
    def test_noncontiguous_fallback(self):
        x=torch.randn(7,3).t(); g=torch.ones(7)
        torch.testing.assert_close(rmsnorm(x,g,fallback=True),reference(x,g))
    def test_fallback_nan_propagation(self):
        x=torch.tensor([[1.,float('nan'),-1.]])
        self.assertTrue(torch.isnan(rmsnorm(x,torch.ones(3),relu2=True,fallback=True)).all())
    def test_fallback_empty(self):
        x=torch.empty(0,7,requires_grad=True); g=torch.ones(7,requires_grad=True)
        dx,dg=torch.autograd.grad(rmsnorm(x,g,fallback=True).sum(),(x,g))
        self.assertEqual(dx.shape,x.shape); self.assertEqual(dg.count_nonzero().item(),0)
    def test_analytic_rms_backward_contract(self):
        self._analytic(False)
    def test_analytic_fused_backward_contract(self):
        self._analytic(True)
    def _analytic(self,fused):
        torch.manual_seed(711)
        for d in (7,8192,32768):
            x=torch.randn(2,d,dtype=torch.float64,requires_grad=True)
            g=torch.randn(d,dtype=torch.float64,requires_grad=True)
            dy=torch.randn_like(x); eps=1e-6
            h=x.relu().square() if fused else x
            r=torch.rsqrt(h.square().mean(-1,keepdim=True)+eps)
            y=h*r*g
            expected=torch.autograd.grad(y,(x,g),dy)
            a=dy*g
            dx=r*(a-h*(a*h).mean(-1,keepdim=True)*r.square())
            if fused: dx=dx*2*x.relu()
            dg=(dy*h*r).sum(0)
            for actual,want in zip((dx,dg),expected):
                torch.testing.assert_close(actual,want,rtol=1e-10,atol=1e-10)

if __name__=='__main__': unittest.main(verbosity=2)
