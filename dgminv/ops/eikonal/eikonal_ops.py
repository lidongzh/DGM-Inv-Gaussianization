import torch
import torch.nn as nn
import numpy as np 
from torch.utils.cpp_extension import load
# import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import dgminv
abs_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(abs_path, 'src')
build_path = os.path.join(src_path, 'build')
eigen_path = os.path.join(dgminv.__path__[0], 'lib/eigen-3.3.7')
os.makedirs(build_path, exist_ok=True)
# os.makedirs('./build/', exist_ok=True)

def load_eikonal(src_path):
    eigen_rel_path = os.path.relpath(eigen_path, os. getcwd())
    print(eigen_rel_path)
    eikonal = load(name="eikonal",
            sources=[src_path+'/eikonal.cpp'],
            extra_cflags=[
                '-O3 -lpthread'
            ],
            extra_include_paths=[eigen_rel_path],
            build_directory=build_path,
            verbose=True)
    return eikonal

eikonal_ops = load_eikonal(src_path)

def eikonal(f, srcz, srcx, h):
    return EikonalFunction.apply(f, srcz, srcx, h)
class EikonalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, srcz, srcx, h):
        ctx.in_type = f.dtype
        ctx.device = f.device
        f = f.type(torch.float64).cpu()
        u = eikonal_ops.forward(f, srcx, srcz, h).type(torch.float64)
        ctx.states = (u, f, srcz, srcx, h)
        return u.type(ctx.in_type).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_u):
        grad_u = grad_u.type(torch.float64).cpu()
        u, f, srcz, srcx, h = ctx.states
        out = eikonal_ops.backward(grad_u, u, f, srcx, srcz, h)
        return out[0].type(ctx.in_type).to(ctx.device), None, None, None
