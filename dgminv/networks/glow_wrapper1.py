import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import copy, sys, collections
sys.path.append('../../')
from dgminv.utils.common import img_to_patches, patches_to_img, img_scale, scale_down_bits, scale_up_bits, roll_tensor_batch, calc_z_shapes
from dgminv.ops.yeojohnson import yeojohnson_transform as yj_transform
from scipy import stats
from dgminv.ops.lambert_transform import lambert_transform
from dgminv.ops.ica import FastIca
from dgminv.ops.ortho_trans import OrthoHH, OrthoCP
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def normalize(x):
    return (x - torch.mean(x)) / (torch.std(x) + 1e-6)

def spherical(x):
    return x / torch.norm(x) * np.sqrt(np.prod(x.shape))

class GlowWrap(nn.Module):
    '''
    Inversion class for pictures
    '''
    def __init__(self, \
        glow_inv, 
        nh = 128,
        nw = 128,
        nc = 3,
        n_bits = 5,
        lpsize=8, # latent patch size (H and W directions)
        lseed=1,  # latent vector seed
        temp = 0.7, # latent vector temperature
        min_values = [0.0, 0.0, 0.0],
        max_values = [1.0, 1.0, 1.0],
        gtrans=False, 
        ortho=False, 
        if_ica=True, 
        ica_only_whiten=False, 
        if_yj=True, 
        if_lambt=True, 
        if_standard=True,
        if_spherical=False):
        super().__init__()
        self.Bounds = {}
        
        self.glow_inv = glow_inv
        self.nc, self.nh, self.nw, self.temp = nc, nh, nw, temp
        self.lpsize = lpsize
        self.n_bits = n_bits
        self.min_values, self.max_values = min_values, max_values
        self.gtrans, self.ortho = gtrans, ortho
        if if_spherical: if_standard = False
        self.if_ica, self.if_yj, self.if_lambt, self.if_standard = if_ica, if_yj, if_lambt, if_standard
        self.if_spherical = if_spherical
        if gtrans and ortho:
            raise ValueError('Cannot use G transformation and orthogoal reparam at the same time.')
        if gtrans:
            self.fastica = FastIca(lpsize*lpsize*nc, only_whiten=ica_only_whiten)
        self.set_latent(lseed)

    def get_latent(self):
        return self.latent * 1.0

    def set_latent(self, lseed, device=torch.device('cuda')):
        rs = np.random.RandomState(lseed)
        # temp = torch.from_numpy(rs.randn(1, self.nc, self.nh, self.nw)).type(torch.float32).to(device)
        self.z_shapes = calc_z_shapes(self.nc, self.nw, self.glow_inv.n_flow, self.glow_inv.n_block)
        # print(f'z shapes = {self.z_shapes}')
        z_list = []
        for z_shape in self.z_shapes:
            z_list.append(torch.from_numpy(rs.randn(*z_shape)).type(torch.float32).unsqueeze(0).to(device))
        temp = self.glow_inv.unsqueeze(z_list)

        if self.ortho:
            self.orthotrans = OrthoCP(self.lpsize*self.lpsize*self.nc).to(device)
            self.register_buffer('latent', temp)
        else:
            self.latent = nn.Parameter(temp)

    def gaussianize(self, z):
        # input: z: Tensor, latent tensor, [1, C, H, W]
        z = img_to_patches(z, self.lpsize, self.lpsize, \
            self.nc).contiguous()
        z_shape = z.shape
        z = z.view(z.shape[0], -1)
        if self.if_ica:
            z = self.fastica(z)
        if self.if_yj:
            z = yj_transform(z)
        if self.if_lambt:
            z = lambert_transform(z)
        if self.if_standard:
            z = normalize(z)
        elif self.if_spherical:
            z = spherical(z)
        z = z.view(z_shape)
        z = patches_to_img(z, self.nh, self.nw, self.lpsize, self.lpsize, self.nc)
        return z
    
    def orthogonalize(self, z):
        print(f'z size = {z.shape}')
        z = img_to_patches(z, self.lpsize, self.lpsize, \
            self.nc).contiguous()
        z_shape = z.shape
        z = z.view(z.shape[0], -1)
        z = self.orthotrans(z)
        z = z.view(z_shape)
        z = patches_to_img(z, self.nh, self.nw, self.lpsize, self.lpsize, self.nc)
        return z

    def forward(self):
        print('forward')
        if self.gtrans:
            z = self.gaussianize(self.latent)
            # # do rolling before another layer and shift it back
            # z = roll_tensor_batch(self.latent, mode='forward')
            # z = self.gaussianize(z)
            # z = roll_tensor_batch(z, mode='backward')
            # z = self.gaussianize(z)
        elif self.ortho:
            z = self.orthogonalize(self.latent)
        else:
            z = self.latent
        print(f'patchesLatent norm = {z.reshape(-1).norm()}')
        img, logdet, log_p = self.glow_inv(z * self.temp)
        img = torch.clamp((img + 0.5)*255.0, 0, 255)

        return img, logdet, log_p

if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    import imageio as io
    sys.path.append('../../')
    from dgminv.networks.models_cond_glow import GlowInv
    network_path = './weights/Flow_net_epoch850.pt'
    os.makedirs('./figures', exist_ok=True)
    device = torch.device('cuda')
    print(network_path)
    
    state_dict = torch.load(network_path, map_location=torch.device('cuda'))
    # print(state_dict)
    glow_inv = GlowInv(3, 32, 4, \
        affine=False, conv_lu=True, cond=False)
    glow_inv.eval()
    glow_inv.load_state_dict(state_dict, strict=False)
    for name, param in glow_inv.named_parameters():
        param.requires_grad = False
    glow_inv.to(device)

    model_inv = GlowWrap(glow_inv, lseed=1, gtrans=True, if_ica=True, if_yj=False, if_lambt=False)
    model_inv.to(device)
    for name, param in model_inv.named_parameters():
        if param.requires_grad == True:
            print(name)

    with torch.no_grad():
        img,_,_ = model_inv()
        print(img.max())
        print(img.min())
        io.imsave('./figures/glow_img.jpg', img[0,...].cpu().numpy().transpose(1,2,0))

