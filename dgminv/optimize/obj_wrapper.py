import torch
import torch.nn.functional as F
import math
import numpy as np
from scipy import optimize
from functools import reduce
from collections import OrderedDict
import matplotlib.pyplot as plt
import sys

class PyTorchObjective:
    '''
    PyTorch objective function, wrapped to be called by scipy.optimize. 
    Heavily inspired by https://gist.github.com/gngdb/a9f912df362a85b37c730154ef3c294b
    '''
    def __init__(self, loss, *obj_list, retain_graph=False):
        self.obj_list = obj_list # some pytorch module, that produces a scalar loss
        self.loss = loss
        self.n_objs = len(obj_list)
        # make an x0 from the parameters in this module
        self.original_parameters = [OrderedDict(self.obj_list[i].named_parameters()) for i in range(self.n_objs)]
        self.parameters = [OrderedDict() for _ in  range(self.n_objs)]
        # get rid of parameters not to be inverted for
        for ind in range(self.n_objs):
            for key, value in self.original_parameters[ind].items():
                if value.requires_grad == True:
                    self.parameters[ind][key] = value
        # list of dictionaries. Each dict store the size of parameters in that obj
        self.param_shapes = [OrderedDict() for _ in range(self.n_objs)]
        for ind in range(self.n_objs):
            self.param_shapes[ind] = {key:self.parameters[ind][key].size() for key in self.parameters[ind]}

        # ravel and concatenate all parameters to make x0
        x0 = []
        for ind in range(self.n_objs):
            for key in self.parameters[ind]:
                x0.append(self.parameters[ind][key].data.cpu().numpy().ravel())
        self.x0 = np.concatenate(x0).astype(np.float64)
        
        # if there is at least one object has bounds, we need to pack bounds
        get_bounds = False 
        for ind in range(self.n_objs):
            get_bounds = get_bounds or self.obj_list[ind].Bounds != {}
        if get_bounds:
                self.bounds = self.pack_bounds()
        else:
            self.bounds = None
        self.retain_graph = retain_graph

    def unpack_parameters(self, x):
        """optimize.minimize will supply 1D array, chop it up for each parameter."""
        i = 0

        named_parameters = [OrderedDict() for _ in  range(self.n_objs)]

        for ind in range(self.n_objs):
            for key in self.param_shapes[ind]:
                # multiply all dimensions
                param_len = reduce(lambda x,y: x*y, self.param_shapes[ind][key])
                # slice out a section of this length
                param = x[i:i+param_len]
                # reshape according to this size, and cast to torch
                param = param.reshape(*self.param_shapes[ind][key])
                named_parameters[ind][key] = torch.from_numpy(param)
                # update index
                i += param_len
        return named_parameters

    def pack_grads(self):
        """pack all the gradients from the parameters in the module into a
        numpy array."""
        grads = []
        for ind in range(self.n_objs):
            for p,v in self.parameters[ind].items():
                # print(p)
                grad = v.grad.data.cpu().numpy()
                grads.append(grad.ravel())
        return np.concatenate(grads).astype(np.float64)
    
    def pack_bounds(self):
        """set up bounds for L-BFGS-B"""
        lbounds = []
        ubounds = []
        for ind in range(self.n_objs):
            for key in self.param_shapes[ind]:
                if self.obj_list[ind].Bounds == {}:
                    # print(f'key = {key}, shape = {self.param_shapes[ind][key]}')
                    lbound = -np.inf * np.ones(self.param_shapes[ind][key])
                    ubound = np.inf * np.ones(self.param_shapes[ind][key])
                else:
                    lbound, ubound = self.obj_list[ind].Bounds[key][0], self.obj_list[ind].Bounds[key][1]
                lbounds.append(lbound.ravel())
                ubounds.append(ubound.ravel())
        return optimize.Bounds(np.concatenate(lbounds).astype(np.float64), \
            np.concatenate(ubounds).astype(np.float64))

    def is_new(self, x):
        # if this is the first thing we've seen
        if not hasattr(self, 'cached_x'):
            return True
        else:
            # compare x to cached_x to determine if we've been given a new input
            x, self.cached_x = np.array(x), np.array(self.cached_x)
            error = np.abs(x - self.cached_x)
            return error.max() > 1e-8

    def cache(self, x):
        # unpack x and load into module 
        state_dicts = self.unpack_parameters(x)
        for ind in range(self.n_objs):
            # don't forget about buffers
            for name,buf in self.obj_list[ind].named_buffers():
                state_dicts[ind][name] = buf
            self.obj_list[ind].load_state_dict(state_dicts[ind], strict=False)
            # zero the gradient
            self.obj_list[ind].zero_grad()
        self.cached_x = x
        loss_val = self.loss()
        self.f = loss_val.item()
        loss_val.backward(retain_graph=self.retain_graph)
        self.jb = self.pack_grads()
        # print(f'grad length = {self.jb.shape}')

    def fun(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.f

    def jac(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.jb
    

# Nonlinear constraint wrapper for PyTorch
class PyTorchConstraint(PyTorchObjective):
    '''
    PyTorch nonlinear constraint function, wrapped to be called by scipy.optimize. 
    '''
    def __init__(self, constraint, *obj_list, ncon=1, con_obj_id=[0], cons_para_names=(), cl=None, cu=None):
        super().__init__(constraint, *obj_list)
        print(len(obj_list))
        self.ncon = ncon
        self.con_obj_id = con_obj_id
        self.cl, self.cu = np.array(cl), np.array(cu)
        assert(self.cl.size == self.cu.size), 'cl size is not equal to cu size'
        self.cbounds = self.pack_cbounds()
        self.cons_para_names = cons_para_names
    
    def pack_cbounds(self):
        """set up bounds IPOPT"""
        if self.cl.size == 1 and self.cu.size == 1 and self.ncon >= 1:
            self.cl = self.cl * np.ones(self.ncon)
            self.cu = self.cu * np.ones(self.ncon)
        return optimize.Bounds(self.cl.astype(np.float64), self.cu.astype(np.float64))

    def cache(self, x):
        # unpack x and load into module 
        state_dicts = self.unpack_parameters(x)
        for ind in range(self.n_objs):
            # don't forget about buffers
            for name,buf in self.obj_list[ind].named_buffers():
                state_dicts[ind][name] = buf
            self.obj_list[ind].load_state_dict(state_dicts[ind], strict=False)
            # zero the gradient
            self.obj_list[ind].zero_grad()
        self.cached_x = x
        f, jac_list = self.loss()
        # print(f'f dim = {f.shape}, jac dim = {jac.shape}')
        self.f = f.cpu().numpy()
        jac_whole_list = []
        # print(f'jac_list = {jac_list}')
        for i in range(len(jac_list)):
            for ind in range(self.n_objs):
                self.obj_list[ind].zero_grad()
            # print(self.parameters[self.con_obj_id]['patchesLatent'].grad)
            self.parameters[self.con_obj_id][self.cons_para_names].grad = jac_list[i]
            packed_grads = self.pack_grads()
            # print(f'packed_grads = {packed_grads}')
            jac_whole_list.append(packed_grads)
        self.jb = np.stack(jac_whole_list, axis=0)
        # print(self.parameters)
        # print(f'jac_whole_list = {jac_whole_list}')
        # print(f'jac_whole_list[0] = {jac_whole_list[0].cpu().numpy().ravel()[0:50]}')
        # print(f'jac_whole_list[0] shape = {jac_whole_list[0].shape}')
        # print(f'stacked jac = {self.jacobian[:,0:50]}')
        # print(f'con jac in obj_wrapper = {jac_list[0].cpu().numpy().ravel()[0:50]}')
        # sys.exit('pause')

    def fun(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.f # is now an array of dimension m

    def jac(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.jb # is a matrix of dimension m x n