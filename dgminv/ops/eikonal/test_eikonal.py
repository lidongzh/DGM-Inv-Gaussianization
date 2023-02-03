import numpy as np 
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('../../../')
from dgminv.ops.eikonal.eikonal_ops import eikonal
from dgminv.utils.grad_check import GradChecker

v = 0.5*torch.ones(100, 200, dtype=torch.float64)
srcz = 8
srcx = 10
h = 1.0

v[50:80, 90:130] = 2.0
v.requires_grad = True

u = eikonal(v, srcz, srcx, h)
print(f'u = {u}')

plt.figure()
plt.imshow(u.detach().cpu().numpy())
plt.colorbar()

b = torch.sum(u**2)
print(f'loss = {b}')
b.backward()
print(f'grad = {v.grad}')

plt.figure()
plt.imshow(v.grad.detach().cpu().numpy())
plt.show()

def test_eikonal(x):
    return torch.sum(torch.tanh(eikonal(x, srcz, srcx, h)))

gc = GradChecker(test_eikonal, v.detach().clone(), mul=1/10.0, vis=True)
gc.check()
