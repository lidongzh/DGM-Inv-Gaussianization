import numpy as np
import torch
from scipy import special
import sys
sys.path.append('../../')
from dgminv.utils.lambertw import lambertw as lw
from dgminv.utils.common import timeit

# Based on tfp lambertw
def lambertw(z):
    return lambertw_fun.apply(z)
class lambertw_fun(torch.autograd.Function):
    @staticmethod
    @timeit
    def forward(ctx, z):
        ctx.z = z
        ctx.device = z.device 
        ctx.shape = z.shape 
        ctx.dtype = z.dtype 
        tp = np.float32 if ctx.dtype == torch.float32 else np.float64
        z_np = z.detach().clone().cpu().numpy().ravel().astype(tp)
        wz = torch.from_numpy(np.real(special.lambertw(z_np))).view(ctx.shape).type(ctx.dtype).to(ctx.device)
        # wz = lw(z)
        ctx.wz = wz
        return wz
    @staticmethod
    @timeit
    def backward(ctx, grad_output):
        z = ctx.z
        wz = ctx.wz
        # print(torch.min(wz))
        grad_wz = (grad_output * torch.where(z == 0.0,
                        torch.ones_like(wz),
                        (wz / (z * (1. + wz))))).type(ctx.dtype)
        return grad_wz


if __name__ == '__main__': 
    from scipy.special import lambertw as sp_lamw
    import numpy as np
    from dgminv.utils.grad_check import GradChecker

    torch.random.manual_seed(0)
    x = torch.randn(256*256, dtype=torch.float64).abs()

    def func(x):
        return torch.sum(torch.tanh(lambertw(x)))

    gc = GradChecker(func, x, mul=1/10.0, vis=True)
    gc.check()

    
    torch_lamw = lambertw(x)
    scipy_lamw = torch.from_numpy(np.real(sp_lamw(x.detach().numpy()))).float()
    print(torch_lamw[:10], scipy_lamw[:10])
    print((torch_lamw - scipy_lamw).abs().max())
    print(lambertw(torch.ones(1)*1e-8), sp_lamw(1e-8))
    

