import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import scipy.optimize as optimize
from scipy.sparse.linalg import LinearOperator, gmres
import sys
sys.path.append('../../')
from dgminv.optimize.obj_wrapper import PyTorchObjective
from dgminv.utils.common import timeit
torch.manual_seed(1)

# Based on the implementation fo FastICA from scikit-learn:
# https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98/sklearn/decomposition/_fastica.py#L340
class FastIca(nn.Module):
    def __init__(self, n_feature, only_whiten=False):
        super().__init__()
        self.register_buffer('W_init', torch.eye(n_feature))
        self.only_whiten = only_whiten

    def compute_W(self, X, max_iter=None, W_init=None, tol=1e-4):
        # X shape [n_features, n_samples]
        ns = X.shape[1]
        if W_init == None:
            W_init = self.W_init
        W = self.W_init.type(X.dtype)

        W0 = W
        for i in range(max_iter):
            Y = torch.mm(W.T, X)
            gY, g_prime_Y = G_logcosh(Y)

            W = 0.8 * torch.mm(X, gY.T)/ns - g_prime_Y[None,:] * W
            W = self.sym_decorr(W)
            # builtin max, abs are faster than numpy counter parts.
            with torch.no_grad():
                lim = torch.max(torch.abs(torch.abs(torch.diagonal(torch.mm(W0.T, W))) - 1.0))
            W0 = W
            if lim.item() < tol:
                break
        return W


    def sym_decorr(self, W, C=None, max_iter=100, tol=1e-4):
        if C == None:
            C = torch.eye(W.shape[0]).to(W.device).type(W.dtype)

        if W.dtype == torch.float64:
            tol = 1e-8
        W = W / torch.sqrt(torch.norm(torch.mm(W.T, torch.mm(C, W)), p=2))
        W0 = W
        for i in range(max_iter):
            W = 1.5 * W - 0.5 * torch.mm(W, torch.mm(W.T, torch.mm(C, W)))
            with torch.no_grad():
                diff = torch.norm(W - W0)
            if diff.item() < tol:
                return W
            W0 = W
        print('sym_decorr does not converge!')
        return W

    # @timeit
    def whiten(self, X):
        X = X - X.mean(dim=-1, keepdim=True)
        C = (1. - 1e-3) * torch.mm(X, X.T)/(X.shape[1]-1) + 1e-3 * torch.eye(X.shape[0]).to(X.device).type(X.dtype)
        W = self.sym_decorr(self.W_init.type(X.dtype), C)
        X = torch.mm(W.T, X)
        return X
    
    # @timeit
    def zca_whiten(self, X):
        # print('zca whiten')
        x_dtype = X.dtype
        X = X.type(torch.float64)
        X = X - X.mean(dim=-1, keepdim=True)
        C = (1. - 1e-3) * torch.mm(X, X.T)/(X.shape[1]-1) + 1e-3 * torch.eye(X.shape[0]).to(X.device).type(X.dtype)
        L, D = torch.linalg.eigh(C)
        L = 1./torch.sqrt(L)
        U = torch.mm(D, torch.mm(torch.diag(L), D.T))
        return torch.mm(U, X).type(x_dtype)


    def forward(self, X, max_iter=10, W=None):
        # X orig shape [n_sample, n_features]
        assert(X.shape[0] >= X.shape[1]), 'Number of samples should be larger than the dimension!'
        if max_iter == 0:
            return X
        X = X.T
        x_dtype = X.dtype
        X = X.type(torch.float64)
        # X = self.whiten(X)
        # X = self.iter_whiten(X)
        X = self.zca_whiten(X)
        if self.only_whiten:
            return X.T.type(x_dtype)
        self.W = self.compute_W(X, max_iter=max_iter, W_init=W)
        return torch.matmul(self.W.T, X).T.type(x_dtype)

def G_logcosh(x):
    alpha = 1.0
    x = x * alpha
    gx = torch.tanh(x)  # apply the tanh inplace
    g_x = alpha * (1.0 - gx**2)
    return gx, g_x.mean(dim=-1)

def G_exp(x):
    exp = torch.exp(-(x ** 2) / 2)
    gx = x * exp
    g_x = (1 - x ** 2) * exp
    return gx, g_x.mean(dim=-1)

def G_cube(x):
    return x ** 3, (3 * x ** 2).mean(dim=-1)

