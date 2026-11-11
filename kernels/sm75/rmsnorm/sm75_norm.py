"""Standalone FP32 RMSNorm and fused ReLU-squared/RMSNorm for CUDA SM75.

No production patching. First-order autograd only. No TMA, async-copy, BF16,
tensor-core, or recent GPU requirement. Build manually with bash build.sh.
"""
import ctypes
import math
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable

_LIB = None

def _library():
    global _LIB
    if _LIB is None:
        path = Path(__file__).parent / 'build' / 'librmsnorm_sm75.so'
        lib = ctypes.CDLL(str(path))
        p, i64, i, f = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float
        lib.sm75_rms_forward.argtypes = [p,p,p,p,i64,i64,f,i,p]
        lib.sm75_rms_backward.argtypes = [p,p,p,p,p,p,p,i64,i64,i,p]
        lib.sm75_rms_forward.restype = i
        lib.sm75_rms_backward.restype = i
        lib.sm75_error_string.argtypes = [i]
        lib.sm75_error_string.restype = ctypes.c_char_p
        _LIB = lib
    return _LIB

def _check(status):
    if status:
        raise RuntimeError(_library().sm75_error_string(status).decode())

def _validate(x, gamma, eps):
    if not x.is_cuda or not gamma.is_cuda or x.device != gamma.device:
        raise ValueError('x and gamma must be CUDA tensors on the same device')
    if x.dtype != torch.float32 or gamma.dtype != torch.float32:
        raise TypeError('the native path accepts FP32 x and gamma only')
    if x.ndim < 1 or x.shape[-1] <= 0 or gamma.shape != (x.shape[-1],):
        raise ValueError('gamma must have shape [x.shape[-1]] with positive width')
    if not x.is_contiguous() or not gamma.is_contiguous():
        raise ValueError('the native path requires contiguous tensors')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('epsilon must be finite and positive')
    rows, d = x.numel() // x.shape[-1], x.shape[-1]
    if rows > 65535*32 or d > 2147483647:
        raise ValueError('tensor exceeds the bounded launch geometry')
    return rows, d

def reference(x, gamma, eps=1e-6, relu2=False):
    h = x.relu().square() if relu2 else x
    return h * torch.rsqrt(h.float().square().mean(-1,keepdim=True) + eps) * gamma

def forward_out(x, gamma, out, rstd, eps=1e-6, relu2=False):
    """Low-level out variant, no autograd. Buffers must not alias inputs."""
    rows,d = _validate(x,gamma,eps)
    for t,shape in ((out,x.shape),(rstd,(rows,))):
        if t.shape != shape or t.device != x.device or t.dtype != x.dtype or not t.is_contiguous():
            raise ValueError('invalid output/rstd buffer')
    if len({t.untyped_storage().data_ptr() for t in (x,gamma,out,rstd)}) != 4 and rows:
        raise ValueError('buffers may not share storage')
    if not rows: return out
    lib = _library()
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        _check(lib.sm75_rms_forward(x.data_ptr(),gamma.data_ptr(),out.data_ptr(),rstd.data_ptr(),
            rows,d,eps,int(relu2),stream.cuda_stream))
    return out

def backward_raw(x,gamma,dy,rstd,relu2=False):
    rows,d = _validate(x,gamma,1e-6)
    if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype:
        raise ValueError('invalid output gradient')
    if rstd.shape != (rows,) or rstd.device != x.device or rstd.dtype != x.dtype or not rstd.is_contiguous():
        raise ValueError('invalid saved rstd')
    dy = dy.contiguous()
    dx, dg = torch.empty_like(x), torch.empty_like(gamma)
    if not rows:
        dg.zero_()
        return dx,dg
    parts = (rows+31)//32
    partial = torch.empty((parts,d),device=x.device,dtype=x.dtype) if parts>1 else None
    lib = _library()
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        _check(lib.sm75_rms_backward(x.data_ptr(),gamma.data_ptr(),dy.data_ptr(),rstd.data_ptr(),
            dx.data_ptr(),dg.data_ptr(),partial.data_ptr() if partial is not None else None,
            rows,d,int(relu2),stream.cuda_stream))
    return dx,dg

class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gamma,eps,relu2):
        rows,_ = _validate(x,gamma,eps)
        out = torch.empty_like(x)
        rstd = torch.empty((rows,),device=x.device,dtype=x.dtype)
        forward_out(x,gamma,out,rstd,eps,relu2)
        ctx.save_for_backward(x,gamma,rstd)
        ctx.relu2 = relu2
        return out
    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        dx,dg = backward_raw(*ctx.saved_tensors[:2],dy,ctx.saved_tensors[2],ctx.relu2)
        return dx,dg,None,None

def rmsnorm(x,gamma,eps=1e-6,*,relu2=False,fallback=False):
    """Strict FP32 native path; explicit fallback supports other dtype/layout cases."""
    if fallback and (not x.is_cuda or not gamma.is_cuda or x.dtype != torch.float32 or
                     gamma.dtype != torch.float32 or not x.is_contiguous() or not gamma.is_contiguous()):
        return reference(x,gamma,eps,relu2)
    return _Norm.apply(x,gamma,eps,relu2)
