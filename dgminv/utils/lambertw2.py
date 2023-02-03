import torch

# ported from tfp. Not used

def lambertw_winitzki_approx(z):
    log1pz = torch.log1p(z)
    return log1pz * (1. - torch.log1p(log1pz) / (2. + log1pz))


def _lambertw_principal_branch(z, name=None):
    tol = 1e-12 
    # Start while loop with the initial value at the approximate Lambert W
    # solution, instead of 'z' (for z > -1 / exp(1)).  Using 'z' has bad
    # convergence properties especially for large z (z > 5).
    w = torch.where(z > -np.exp(-1.), lambertw_winitzki_approx(z), z)
    for iter in range(10000):
        print(f'iter = {iter}')
        f = w - z * torch.exp(-w)
        delta = f / (w + 1. - 0.5 * (w + 2.) * f / (w + 1.))
        w_next = w - delta
        if torch.any(~(torch.abs(delta) > tol * torch.abs(w_next))):
            break
    
    return w

def lambertw2(z):
    return lambertw_fun.apply(z)
class lambertw_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.z = z
        wz = _lambertw_principal_branch(z)
        ctx.wz = wz.clone()
        return wz 
    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.z 
        wz = ctx.wz
        grad_wz = (grad_output * torch.where(z == 0.0,
                        torch.ones_like(wz),
                        wz / (z * (1. + wz))))
        return grad_wz


if __name__ == '__main__': 
    from scipy.special import lambertw as sp_lamw
    import numpy as np
    import sys
    sys.path.append('../../')
    from dgminv.utils.grad_check import GradChecker

    torch.random.manual_seed(0)
    x = torch.randn(1000).abs()

    def func(x):
        return torch.sum(torch.tanh(lambertw2(x)))

    gc = GradChecker(func, x, mul=1/10.0, vis=True)
    gc.check()

    
    torch_lamw = lambertw2(x)
    scipy_lamw = torch.from_numpy(np.real(sp_lamw(x.numpy()))).float()
    print(torch_lamw[:10], scipy_lamw[:10])
    print((torch_lamw - scipy_lamw).abs().max())
    print(lambertw2(torch.ones(1)*1e-8), sp_lamw(1e-8))

