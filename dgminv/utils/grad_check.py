import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

class GradChecker:
    def __init__(self, fun, x0, mul=0.5, npoints=5, device='cpu', vis=True):
        self.fun = fun
        self.x0 = x0
        if isinstance(self.x0, np.ndarray):
            self.x0 = torch.tensor(x0)
        self.x0 = self.x0.to(device)
        self.x0.requires_grad = True
        self.perturb = torch.rand_like(self.x0)
        self.perturb = self.perturb / torch.norm(self.perturb)
        # print(f'perturb = {self.perturb}')
        self.mul = mul
        self.npoints = npoints
        self.vis = vis
    
    def check(self, save_fig_addr=None):
        f = self.fun(self.x0)
        print(f'fun = {f.item()}')
        f.backward()
        grad = self.x0.grad.clone()
        # print(f'grad = {grad}')
        self.x0.grad = None
        deltas = torch.tensor([pow(self.mul, i) for i in range(self.npoints)])
        print(f'deltas = {deltas}')
        fs = torch.empty(self.npoints)
        fs_fd = torch.empty(self.npoints)
        dot_product = torch.sum(self.perturb.reshape(-1) * grad.reshape(-1))
        print(f'dot_product = {dot_product}')
        with torch.no_grad():
            for i in range(self.npoints):
                fs_fd[i] = self.fun(self.x0 + self.perturb * deltas[i]) - self.fun(self.x0)
                fs[i] = fs_fd[i] - deltas[i] * dot_product
        fs_fd = torch.abs(fs_fd)
        fs = torch.abs(fs)
        print(f'fs = {fs}')
        print(f'fs_fd = {fs_fd}')
        if self.vis:
            plt.rcParams.update({'font.size': 16})
            fs = fs.cpu().numpy()
            fs_fd = fs_fd.cpu().numpy()
            deltas = deltas.cpu().numpy()
            line1 = deltas * fs_fd[0]/deltas[0]*0.1
            line2 = np.power(deltas, 2) * fs[0]/deltas[0]*0.1
            plt.figure(figsize=(8,6))
            plt.loglog(deltas, fs_fd, '-', linewidth=2, label='Finite difference')
            plt.loglog(deltas, fs, '-', linewidth=2, label='AD with custom gradient')
            plt.loglog(deltas, line1, '--', linewidth=2, label='1st order')
            plt.loglog(deltas, line2, '--', linewidth=2, label='2nd order')
            plt.scatter(deltas, fs_fd, s=30, c='C0', marker='*')
            plt.scatter(deltas, fs, s=30, c='C1', marker='*')
            plt.xlabel(r'$\epsilon$')
            plt.ylabel('Approximation error')
            plt.legend()
            if save_fig_addr == None:
                plt.show()
            else:
                plt.savefig(save_fig_addr, bbox_inches='tight', dpi=300)
                plt.close('all')
    
if __name__ == '__main__':
    from dgminv.ops.yeojohnson import yeojohnson_transform as ys
    from dgminv.ops.yeojohnson import yeojohnson_normmax_th as ys_max
    from dgminv.ops.yeojohnson import yeojohnson_llf_th as yllf
    loss = nn.MSELoss()

    x = np.random.randn(1000)
    lmd = 2.0
    th_x = torch.tensor(x).type(torch.float64)
    th_lmd = torch.tensor(lmd).type(torch.float64)

    def fun (x):
        # return torch.sum(F.tanh(ys(x)))
        return torch.sum(torch.tanh(ys_max(x)))
        # return torch.sum(torch.tanh(yllf(x, th_x)))

    # th_x.requires_grad = True
    # # th_lmd.requires_grad = True
    # torch.autograd.gradcheck(fun, th_x, eps=1e-06, atol=1e-05, rtol=0.001, raise_exception=True, check_sparse_nnz=False, nondet_tol=0.0,\
    #     check_undefined_grad=True, check_grad_dtypes=False)

    gc = GradChecker(fun, th_x, mul=1/20.0)
    gc.check()



    


    