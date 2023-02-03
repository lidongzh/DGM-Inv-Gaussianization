# Based on scipy stats, implemented the forward and backward passes
import numpy as np 
import torch
import torch.nn as nn
import sys
sys.path.append('../../')
from dgminv.utils.common import torch_dtype as td
import scipy.optimize as optimize
from scipy.stats import yeojohnson_normmax
from scipy.stats import yeojohnson_llf
from dgminv.optimize.brent_optimizer import brent


def yeojohnson_transform(x):
    """Return x transformed by the Yeo-Johnson power transform with given
    parameter lmbda."""

    assert(isinstance(x, torch.Tensor)), 'x should be a torch tensor!'
    orig_shape = x.shape
    x = x.contiguous().view(-1)
    x_np = x.detach().cpu().numpy().ravel()

    lmbda = yeojohnson_normmax_th(x)

    out = _yeojohnson_transform_th(x, lmbda)
    out = out.view(orig_shape)
    return out

def _yeojohnson_transform_th(x, lmbda):
    out = torch.zeros_like(x)
    pos = x >= 0  # binary mask

    if isinstance(lmbda, torch.Tensor):
        lmbda_val = lmbda.item()
    else:
        lmbda_val = lmbda

    # when x >= 0
    if abs(lmbda_val) < np.spacing(1.):
        out[pos] = torch.log1p(x[pos])
    else:  # lmbda != 0
        out[pos] = (torch.pow(x[pos] + 1, lmbda) - 1) / lmbda

    # when x < 0
    if abs(lmbda_val - 2) > np.spacing(1.):
        out[~pos] = -(torch.pow(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
    else:  # lmbda == 2
        out[~pos] = -torch.log1p(-x[~pos])

    return out

def yeojohnson_normmax_th(x):
    return yeojohnson_normmax_custom.apply(x)
class yeojohnson_normmax_custom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x.detach().clone()
        with torch.no_grad():
            lmbda = yeojohnson_normmax_np(x)
        # lmbda = torch.tensor(yeojohnson_normmax(x.detach().cpu().numpy().ravel())).to(x.device)
        ctx.output = lmbda.detach().clone()
        return lmbda.detach()
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.x
        lmbda = ctx.output
        x.requires_grad = True
        lmbda.requires_grad = True 
        x.grad, lmbda.grad = None, None
        with torch.enable_grad():
            res = yeojohnson_llf_th_grad(lmbda, x)
            res.backward()
        return -(grad_output * (torch.tensor(1.0,dtype=x.dtype)/lmbda.grad) * x.grad).detach()

def yeojohnson_normmax_np(x, brack=(-2, 2)):
    # def _neg_llf(lmbda, data):
    #     lmb = torch.tensor(lmbda, dtype=torch.float64)
    #     d = torch.tensor(data, dtype=torch.float64)
    #     with torch.no_grad():
    #         out = -yeojohnson_llf_th(lmb, d)
    #         return out
    def _neg_llf(lmbda, data):
        return -yeojohnson_llf(lmbda, data)
    x_np = x.detach().cpu().numpy().ravel()
    lmbda = brent(_neg_llf, brack=brack, args=(x_np,), grad_fun=yeojohnson_llf_th_grad)
    return torch.tensor(lmbda, dtype=x.dtype).to(x.device)

def yeojohnson_llf_th(lmb, data):
    n_samples = data.shape[0]

    if n_samples == 0:
        return torch.tensor(np.nan)
    trans = _yeojohnson_transform_th(data, lmb)

    loglike = -n_samples / 2.0 * torch.log(trans.var(axis=0))
    loglike += (lmb - 1.0) * (torch.sign(data) * torch.log(torch.abs(data) + 1.0)).sum(axis=0)

    return loglike

def yeojohnson_llf_th_grad(lmb, data, if_graph=True):
    tp = data.dtype
    if not isinstance(data, torch.Tensor):
        tp = torch.float32 if data.dtype==np.float32 else torch.float64
        data = torch.tensor(data, dtype=tp, requires_grad = True)
        lmb = torch.tensor(lmb, dtype=tp, requires_grad = True)
    l = -yeojohnson_llf_th(lmb, data) 
    return torch.autograd.grad(l, lmb, grad_outputs=torch.tensor(1.0, dtype=tp).to(lmb.device), \
        retain_graph=if_graph, create_graph=if_graph)[0]

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})
    import os
    os.makedirs('./figures', exist_ok=True)
    # plot yeo-johnson
    x = torch.tensor(np.linspace(-4,4,100))

    lmbda = [-1.0, 0.0, 1.0, 2.0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for l in lmbda:
        out = _yeojohnson_transform_th(x, l)
        ax.plot(x, out, label=f"$\lambda$ = {l}", linewidth=3)
        ax.set_aspect(1.5/(10/4.5))
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig('./figures/yeo_johnson.jpg', bbox_inches='tight', dpi=300)
    
    from dgminv.utils.grad_check import GradChecker

    torch.random.manual_seed(0)
    x = torch.randn(256*256, dtype=torch.float64).abs()

    def func(x):
        return torch.sum(torch.tanh(yeojohnson_transform(x)))

    gc = GradChecker(func, x, mul=1/10.0, vis=True)
    gc.check()