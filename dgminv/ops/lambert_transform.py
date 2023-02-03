# Based on https://github.com/gregversteeg/gaussianize/blob/master/gaussianize.py
# Implemented the forward & backward passes

import numpy as np 
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import sys, warnings
sys.path.append('../../')
from dgminv.utils.common import torch_dtype as td
from scipy import special
import scipy.optimize as optimize
from scipy.stats import kurtosis as spk
from scipy.stats import skew as ssk
from dgminv.utils.lambertw import lambertw
from dgminv.utils.kurtosis import kurtosis
from dgminv.utils.skewness import skew
from dgminv.optimize.brent_optimizer import brent

_EPS = 1e-6

def delta_init(z):
    gamma = kurtosis(z, fisher=False, bias=False)
    # with np.errstate(all='ignore'):
    delta0 = torch.clip(1. / 66 * (torch.sqrt(66 * gamma - 162.) - 6.), 0.01, 0.48)
    if not torch.isfinite(delta0):
        delta0 = torch.tensor(0.01, dtype=z.dtype)
    # assert(delta0.dtype == torch.float64), 'needs double'
    return delta0


def igmm(y, tol=1e-5, max_iter=100):
    # Infer mu, sigma, delta using IGMM in Alg.2, Appendix C
    if y.dtype == torch.float64:
        tol = 1e-8
    delta0 = delta_init(y)
    tau1 = (torch.mean(y).to(y.device), torch.std(y).to(y.device) * (1. - 2. * delta0) ** 0.75, delta0)
    for k in range(max_iter):
        # print(f'iter = {k}')
        tau0 = tau1
        z = (y - tau1[0]) / (tau1[1] + 1e-6)
        # print('z = ', z)
        delta1 = delta_gmm_th(z)
        # print('delta 1 =', delta1)
        # delta1 = torch.tensor(0.2, dtype=torch.float64)
        if delta1.item() == 0:
            break
        x = tau1[0] + tau1[1] * w_d_th(z, delta1)
        mu1, sigma1 = torch.mean(x), torch.std(x)
        tau1 = (mu1, sigma1, delta1)
        # print(f'tau1 = {tau1}')
        with torch.no_grad():
            if torch.norm(torch.tensor(tau1) - torch.tensor(tau0)) < tol:
                print(f'igmm iter = {k}')
                break
            else:
                if k == max_iter - 1:
                    warnings.warn("Warning: No convergence after %d iterations. Increase max_iter." % max_iter)
    return tau1


# ==================== optimize for delta =========================
def inside_func_th(q, z):
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=z.dtype, requires_grad = True)
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=z.dtype, requires_grad = True)
    u = w_d_th(z, q)
    if not torch.all(torch.isfinite(u)):
        print('inside func th u not finite')
        return torch.tensor(0.0, dtype=z.dtype)
    else:
        k = kurtosis(u, fisher=True, bias=False)**2
        if not torch.isfinite(k) or k.item() > 1e10:
            return torch.tensor(1e10, dtype=z.dtype)
        else:
            return k

def inside_func_th_grad(q, z, if_graph=True):
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=z.dtype, requires_grad = True)
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=z.dtype, requires_grad = True)
    l = inside_func_th(q, z).unsqueeze(0)
    grad =  torch.autograd.grad(l, q, grad_outputs=torch.tensor([1.0], dtype=z.dtype).to(q.device), \
        retain_graph=if_graph, create_graph=if_graph)[0]
    return grad


def delta_gmm_np(z):
    # Alg. 1, Appendix C
    delta0 = delta_init(z).detach().cpu().numpy()
    # delta0 = delta_init(z)
    z_np = z.detach().cpu().numpy().ravel()
    # delta0 = delta_init(z_np)

    def inside_func(q,z_np):
        # u = w_d_th(torch.from_numpy(z_np), torch.tensor(q, dtype=torch.float64)).detach().numpy()
        u = w_d(z_np, q)
        if not np.all(np.isfinite(u)):
            print('return 0')
            return 0.
        else:
            k = spk(u, fisher=True, bias=False)**2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            else:
                return k

    def _print(xk):
        print(f'xk = {xk}')
    res = optimize.minimize(inside_func, delta0, args=(z_np,),  method='L-BFGS-B', bounds=((1e-6, np.inf),), tol=None, callback=None, options={'disp': False, 'iprint': 101, \
    'gtol': 1e-12, 'maxiter': 100, 'ftol': 1e-12, 'maxcor': 30, 'maxfun': 15000})
    res = res.x

    # with torch.enable_grad():
    #     print(inside_func_th_grad(res, z_np))
    return torch.tensor(res, dtype=z.dtype, device=z.device)



def delta_gmm_th(z):
    return delta_gmm_custom.apply(z)
class delta_gmm_custom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.z = z.detach().clone()
        # with torch.no_grad():
        delta = delta_gmm_np(z)
        # lmbda = torch.tensor(yeojohnson_normmax(x.detach().cpu().numpy().ravel())).to(x.device)
        ctx.output = delta.detach().clone()
        # print('lmbda = ', lmbda)
        # print(f'ctx output = {ctx.output}')
        return delta
    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.z
        delta = ctx.output
        z.requires_grad = True
        delta.requires_grad = True 
        z.grad, delta.grad = None, None
        with torch.enable_grad():
            # print(f'backward delta = {delta}')
            res = inside_func_th_grad(delta, z)
            # print(f'res = {res}')
            res.backward()
        # print(f'delta = {delta}')
        # print(np.any(np.isnan(z.detach().cpu().numpy())))
        # print(f'z ={z}')
        # print(f'delta grad = {delta.grad}')
        # print(f'input grad = {torch.max(z.grad)}')
        return -(grad_output * (torch.tensor(1.0,dtype=z.dtype)/delta.grad) * z.grad).detach()

# ============================================================


def lambert_transform(x):
    """Return x transformed by the Lambert transform."""

    assert(isinstance(x, torch.Tensor)), 'x should be a torch tensor!'
    orig_shape = x.shape
    orig_dtype = x.dtype
    # x = x.contiguous().view(-1)
    # x = x.type(torch.float64)
    x = x.flatten()
    x_np = x.detach().cpu().numpy().ravel()

    kur = spk(x_np, fisher=False)
    if (kur > 3.0):
        tau = igmm(x)
        # print(f'tau = {tau}')
        out = w_t_th(x, tau)
    else:
        # print(f'kur = {kur}')
        out = x
    # tau = igmm(x)
    # out = w_t_th(x, tau)

    out = out.view(orig_shape)
    return out.type(orig_dtype)

def w_t_th(y, tau):
    return tau[0] + tau[1] * w_d_th((y - tau[0]) / tau[1], tau[2])

def w_d_th(z, delta):
    wd = torch.sign(z) * torch.sqrt(lambertw(delta * z ** 2) / delta)
    return wd

def w_d(z, delta):
    # Eq. 9
    return np.sign(z) * np.sqrt(np.real(special.lambertw(delta * z ** 2)) / delta)

if __name__ == '__main__':
    from dgminv.utils.grad_check import GradChecker

    torch.random.manual_seed(0)
    x = torch.randn(256*256, dtype=torch.float64).abs()

    def func(x):
        return torch.sum(torch.tanh(lambert_transform(x)))

    gc = GradChecker(func, x, mul=1/10.0, vis=True)
    gc.check()