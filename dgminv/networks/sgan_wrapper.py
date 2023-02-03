import numpy as np
import sys
import torch
sys.path.append('../../')
sys.path.append('../../stylegan2-ada-pytorch')
import dnnlib
import legacy
from collections import OrderedDict
from dgminv.utils.common import img_to_patches, patches_to_img
import copy
from scipy.stats import kurtosis as spk
from einops import rearrange, repeat
from dgminv.ops.yeojohnson import yeojohnson_transform as yj_transform
from dgminv.ops.lambert_transform import lambert_transform
from dgminv.ops.ica import FastIca
from dgminv.ops.ortho_trans import OrthoHH, OrthoCP

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def delattr_force(module, name):
    name_list = name.split('.')
    v = module
    for i in range(len(name_list)-1):
        v = getattr(v, name_list[i])
    delattr(v, name_list[-1])

def setattr_force(module, name, value):
    name_list = name.split('.')
    v = module
    for i in range(len(name_list)-1):
        v = getattr(v, name_list[i])
    setattr(v, name_list[-1], value)

def normalize(x):
    return (x - torch.mean(x)) / (torch.std(x) + 1e-6)

def spherical(x):
    print('spherical')
    return x / torch.norm(x) * np.sqrt(np.prod(x.shape))

class StyleGAN2Wrap(torch.nn.Module):
    def __init__(self, 
        network_pkl, 
        style_psize=8, 
        noise_psize=8, 
        style_seed=1, 
        noise_seed=1, 
        mode='mode-z', 
        noise_update=True, 
        gtrans_style=False, 
        ortho_style=False,
        gtrans_noise=False,
        ortho_noise=False,
        if_ica=True,
        ica_only_whiten=False,
        if_yj=True,
        if_lambt=True,
        if_standard=True,
        if_spherical=False
        ):
        super().__init__()
        device = torch.device('cuda')
        self.device = device
        with dnnlib.util.open_url(network_pkl) as fp:
            self.G = legacy.load_network_pkl(fp)['G_ema'].eval().requires_grad_(False).to(device) # type: ignore
        self.noise_bufs_orig = OrderedDict({ name: buf for (name, buf) in self.G.synthesis.named_buffers() if 'noise_const' in name })

        noise_psize = pair(noise_psize)
        self.noise_psize, self.style_psize = noise_psize, style_psize
        self.noise_update = noise_update
        self.mode = mode
        self.style_seed, self.noise_seed = style_seed, noise_seed
        if if_spherical: if_standard = False
        self.if_ica, self.ica_only_whiten, self.if_yj, self.if_lambt, self.if_standard = \
            if_ica, ica_only_whiten, if_yj, if_lambt, if_standard
        self.if_spherical = if_spherical

        self.Bounds = {}

        # Gaussianization or Orthogonalization
        assert not (if_standard and if_spherical)
        if (not noise_update) and gtrans_noise:
            raise ValueError('Cannot use G transformation but do not update noise.')

        if gtrans_noise and ortho_noise:
            raise ValueError('Cannot use G transformation and orthogoal reparam at the same time.')
        if gtrans_style and ortho_style:
            raise ValueError('Cannot use G transformation and orthogoal reparam at the same time.')
        self.gtrans_noise, self.gtrans_style = gtrans_noise, gtrans_style
        self.ortho_noise, self.ortho_style = ortho_noise, ortho_style

        if gtrans_noise:
            self.fastica_noise = FastIca(noise_psize[0]*noise_psize[1], only_whiten=ica_only_whiten)
        if gtrans_style:
            self.fastica_style = FastIca(style_psize, only_whiten=ica_only_whiten)
        
        self.set_latent(style_seed, noise_seed)

    
    def set_latent(self, style_seed, noise_seed):
        rs= np.random.RandomState(noise_seed)
        noise_psize, style_psize = self.noise_psize, self.style_psize
        self.neglected_noise_patches = []
        self.selected_noise_patches_list = []
        num_noise_patches = 0 # total number of noise patches
        self.buf_to_patch_info = [] # store necessary info to reconstruct noise_img from patches
        self.neglected_patch_info = []
        device = self.device
        for buf in self.noise_bufs_orig.values():
            if buf.shape[0] >= noise_psize[0] and buf.shape[1] >= noise_psize[1]:
                num_noise_patches += (buf.shape[1] // noise_psize[1]) * (buf.shape[0] // noise_psize[0])
                noise_img = torch.from_numpy(rs.randn(*buf.shape)).unsqueeze(0).unsqueeze(0).type(buf.dtype).to(device)
                # print(f'noise_img shape = {noise_img.shape}')
                noise_patches = img_to_patches(noise_img, noise_psize[0], noise_psize[0], 1)
                # print(noise_patches.shape)
                self.selected_noise_patches_list.append(noise_patches)
                # the noise buffers have a square shape
                self.buf_to_patch_info.append((noise_patches.shape[0], buf.shape[0]))
            else:
                self.neglected_patch_info.append(buf.shape[0]*buf.shape[0])
                # self.neglected_noise_patches.append(torch.nn.Parameter(torch.from_numpy(rs.randn(*buf.shape)).type(buf.dtype).to(device)))
        
        noise_dim = noise_psize[0] * noise_psize[1]
        assert noise_dim <= num_noise_patches, 'The dimension of noise patches should not be greater than the numbers'
        self.noise_patches = torch.nn.Parameter(torch.cat(self.selected_noise_patches_list, dim=0))
        # print(f'noise_patches shape = {noise_patches.shape}')
        assert self.noise_patches.shape[0] == num_noise_patches
        # print(f'noise patches shape = {self.noise_patches.shape}')
        # print(f'buf_to_patch_info = {self.buf_to_patch_info}')
        self.flat_neglected_noise = torch.nn.Parameter(torch.from_numpy(rs.randn(sum(self.neglected_patch_info))).type(buf.dtype).to(device))

        if not self.noise_update:
            self.flat_neglected_noise.requires_grad = False
            self.noise_patches.requires_grad = False
        
        # print(self.neglected_noise_patches)
        mode = self.mode
        if mode == 'mode-z' or mode == 'mode-z-':
            num_ws = 1
            self.w = torch.nn.Parameter(torch.from_numpy(np.random.RandomState(style_seed).randn(num_ws, \
                self.G.z_dim)).to(buf.dtype).to(device))
        elif mode == 'mode-z+':
            num_ws = self.G.num_ws
            self.w = torch.nn.Parameter(torch.from_numpy(np.random.RandomState(style_seed).randn(num_ws, \
                self.G.z_dim)).to(buf.dtype).to(device))
            setattr_force(self.G.mapping, 'num_ws', 1)

        elif mode == 'mode-w':
            num_ws = self.G.num_ws
            # self.w = torch.nn.Parameter(torch.from_numpy(np.random.RandomState(style_seed).randn(1, \
            #     num_ws, self.G.z_dim)).to(buf.dtype).to(device))
            self.w = torch.nn.Parameter(torch.zeros(1, num_ws, self.G.z_dim).to(buf.dtype).to(device))
        else:
            raise NotImplementedError

        if self.ortho_noise:
            # self.orthotrans_noise = OrthoHH(noise_psize[0]*noise_psize[1], seed=noise_seed+style_seed).to(device)
            self.orthotrans_noise = OrthoCP(noise_psize[0]*noise_psize[1]).to(device)
            # self.flat_neglected_noise.requires_grad = False
            self.noise_patches.requires_grad = False
        if self.ortho_style:
            # self.orthotrans_style = OrthoHH(style_psize, seed=noise_seed+style_seed).to(device)
            self.orthotrans_style = OrthoCP(style_psize).to(device)
            self.w.requires_grad = False
        
        self.w_to_patches = lambda x: rearrange(x, 'nws (npatch psize) -> (nws npatch) psize', psize=style_psize)
        self.patches_to_w = lambda x: rearrange(x, '(nws npatch) psize -> nws (npatch psize)', nws=num_ws, psize=style_psize)

    def get_latent(self):
        assert self.mode == 'mode-z+'
        assert not (self.ortho_style or self.ortho_noise)
        latent_list = [self.w * 1.0, self.noise_patches * 1.0, self.flat_neglected_noise * 1.0]
        return latent_list

    def noise_patches_to_bufs(self, noise_patches):
        start = 0
        noise_bufs = []
        # for buf in self.neglected_noise_patches:
        #     noise_bufs.append(buf)
        for ndim in self.neglected_patch_info:
            d = int(np.sqrt(ndim))
            noise_bufs.append(self.flat_neglected_noise[start:start+ndim].view(d, d))
            start = start + ndim

        start = 0
        for (count, bufshape) in self.buf_to_patch_info:
            noise_chunk = noise_patches[start:start+count,...]
            # print(f'start:{start}, count:{count}')
            # print(f'noise_chunk shape = {noise_chunk.shape}')
            temp = patches_to_img(noise_chunk, bufshape, bufshape, self.noise_psize[0], self.noise_psize[0], 1)
            noise_bufs.append(temp.view(temp.shape[2], temp.shape[3]))
            start = start + count
        count = 0
        # print(noise_bufs)
        for name, buf in self.noise_bufs_orig.items():
            # delete the original buffer and place the new tensor there
            delattr_force(self.G.synthesis, name)
            setattr_force(self.G.synthesis, name, noise_bufs[count])
            count += 1
        


    def gaussianize_noise(self, z):
        z_shape = z.shape
        z = z.view(z.shape[0], -1)
        if self.if_ica:
            z = self.fastica_noise(z)
        if self.if_yj:
            z = yj_transform(z)
        if self.if_lambt:
            z = lambert_transform(z)
        if self.if_standard:
            z = normalize(z)
        elif self.if_spherical:
            z = spherical(z)
        return z.view(z_shape)
    
    def gaussianize_style(self, z):
        z = self.w_to_patches(z)
        if self.if_ica:
            z = self.fastica_style(z)
        if self.if_yj:
            z = yj_transform(z)
        if self.if_lambt:
            z = lambert_transform(z)
        if self.if_standard:
            z = normalize(z)
        elif self.if_spherical:
            z = spherical(z)
        z = self.patches_to_w(z)
        return z
    
    def orthogonal_noise(self, z):
        z_shape = z.shape
        z = z.view(z.shape[0], -1)
        z = self.orthotrans_noise(z)
        z = z.view(z_shape)
        return z

    def orthogonal_style(self, z):
        z = self.w_to_patches(z)
        z = self.orthotrans_style(z)
        z = self.patches_to_w(z)
        return z

    def forward(self):
        if self.gtrans_noise:
            z = self.gaussianize_noise(self.noise_patches)
        elif self.ortho_noise:
            z = self.orthogonal_noise(self.noise_patches)
        else:
            z = self.noise_patches
        
        if self.gtrans_style:
            w_style = self.gaussianize_style(self.w)
        elif self.ortho_style:
            w_style = self.orthogonal_style(self.w)
        else:
            w_style = self.w
        
        if self.mode != 'mode-z-':
            self.noise_patches_to_bufs(z)
        if self.mode == 'mode-z' or self.mode == 'mode-z-':
            img = self.G(w_style, None, truncation_psi=1, noise_mode='const') 
        elif self.mode == 'mode-z+':
            ws = self.G.mapping(w_style, None, truncation_psi=1).squeeze(1).unsqueeze(0)
            img = self.G.synthesis(ws, noise_mode='const')
        elif self.mode == 'mode-w':
            img = self.G.synthesis(self.w, noise_mode='const')
        else:
            raise NotImplementedError

        return img