import numpy as np
import torch 
import torch.nn as nn
import sys
sys.path.append('../../')
from dgminv.ops.ica import FastIca
from dgminv.utils.grad_check import GradChecker

fica = FastIca(64)

def zca_whitening(X):
    X = X - X.mean(dim=-1, keepdim=True)
    C = torch.mm(X, X.T)/(X.shape[1]-1) + 1e-3 * torch.eye(X.shape[0]).to(X.device).type(X.dtype)
    L, D = torch.linalg.eigh(C)
    L = 1./torch.sqrt(L)
    U = torch.mm(D, torch.mm(torch.diag(L), D.T))
    return torch.mm(U, X)

# X = torch.rand(64, 256).type(torch.float64)

# Y = fica.zca_whiten(X)


# print(torch.mm(Y, Y.T)/(Y.shape[1]-1))

# Y2 = fica.whiten(X)
# print(torch.mm(Y2, Y2.T)/(Y2.shape[1]-1))

# # print(Y - Y2)
# print(torch.max(torch.abs(Y - Y2)))
# assert torch.max(torch.abs(Y - Y2)) < 1e-8

def fun(x):
    x = x.reshape(64, 256)
    return torch.sum(fica.zca_whiten(x).ravel()**2)
    # return torch.sum(fica(x).ravel()**2)
    # return torch.sum(zca_whitening(x).ravel()**2)

th_x = torch.randn(64, 256).type(torch.float64)
gc = GradChecker(fun, th_x, mul=1/10.0, vis=True)
gc.check()