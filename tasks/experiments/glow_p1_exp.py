import os, sys
import matplotlib 
# matplotlib.use ('Agg')
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
import numpy as np 
plt.rcParams.update({'font.size': 40}) 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 
import sys
sys.path.append('../../')
from dgminv.networks.models_cond_glow import GlowInv
from dgminv.networks.glow_wrapper1 import GlowWrap
from dgminv.utils.common import calc_z_shapes
from scipy import stats 
from dgminv.ops.ortho_trans import OrthoCP
torch.manual_seed(0) 
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def normalize (x):
    return (x - torch.mean (x)) / (torch.std(x) + 1e-6)

def spherical (x):
    return x / torch. norm(x) * np.sqrt (np.prod (x.shape))

class GlowWrapExp(GlowWrap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_latent(self, lseed, z=None, device=torch.device('cuda')):
        if isinstance(z, torch.Tensor):
            print('z is a tensor')
            self.latent = nn. Parameter(z.type(torch. float32).to (device))
            return
        rs = np.random.RandomState(lseed)
        # temp = torch.from_numpy(rs.rand(1, self.nc, self.nh, self.nw)).type(torch.float32).to(device)
        self.z_shapes = calc_z_shapes(self.nc, self.nw, self.glow_inv.n_flow, self.glow_inv.n_block)
        # print(f'z shapes = {self.z shapes}')
        z_list = []
        for z_shape in self.z_shapes:
            z_list.append(torch.from_numpy(rs.randn(*z_shape)).type(torch.float32).unsqueeze(0).to(device))
        temp = self.glow_inv.unsqueeze(z_list)
        if self.ortho:
            self.orthotrans = OrthoCP(self.lpsize*self.lpsize*self.nc).to(device)
            self.register_buffer('latent', temp) 
        else:
            self.latent = nn.Parameter(temp)

    def forward(self):
        print('forward' )
        if self.gtrans:
            z = self.gaussianize(self.latent)
        elif self.ortho:
            z = self.orthogonalize(self.latent)
        else:
            z = self.latent
        img, logdet, log_p = self.glow_inv(z * self.temp)
        img = torch.clamp((img + 0.5)*255.0, 0, 255)
        return img, logdet, log_p, z * self.temp
    
nh, nw = 128, 128 # size of images
lpsize = 8 # size of patch size for the ICA layer
network_path = '../glow/training/checkpoints/Flow_net_epoch850.pt'
os.makedirs('./figures_nf', exist_ok=True)
device = torch.device('cuda')

state_dict = torch.load(network_path, map_location=device)
glow_inv = GlowInv(3, 32, 4, affine=False, conv_lu=True, cond=False)
glow_inv.eval()
glow_inv.load_state_dict(state_dict, strict=True)
for name, param in glow_inv.named_parameters():
    param.requires_grad = False
glow_inv.to(device)

from scipy.stats import norm
gaudata = np.random.normal(0, 0.7, 1000)
mu, std = norm.fit(gaudata)
xmin, xmax = -5, 5
def exp_map(z_latent, file_name, **args):
    model_inv = GlowWrapExp(glow_inv, nh=nh, nw=nw, **args)
    model_inv.to(device)
    model_inv.set_latent(0, z=z_latent)

    with torch.no_grad():
        img_reverse, logdet, logp, z_out = model_inv()

    plt.figure(figsize=(10,5))
    plt.imshow(img_reverse.cpu().numpy()[0,...].transpose(1,2,0)/255)
    plt.axis('off')
    plt.savefig('./figures_nf/' + file_name +'.jpg', bbox_inches = 'tight', dpi=300)

    plt.figure()
    stats.probplot(z_latent.detach().cpu().numpy().ravel(), dist=stats.norm, plot=plt)
    plt.savefig('./figures_nf/qq_' + file_name + '_before.jpg', bbox_inches='tight', dpi=300)

    plt.figure()
    stats.probplot(z_out.detach().cpu().numpy().ravel(), dist=stats.norm, plot=plt)
    plt.savefig('./figures_nf/qq_' + file_name + '_after.jpg', bbox_inches='tight', dpi=300)

    plt.figure()
    plt.hist(z_latent.detach().cpu().numpy().ravel(), bins=100, density=True) 
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.savefig('./figures_nf/hist_' + file_name + '_before.jpg', bbox_inches='tight', dpi=300)

    plt.figure()
    plt.hist(z_out.detach().cpu().numpy().ravel(), bins=100, density=True)
    x = np.linspace (xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot (x, p, 'k', linewidth=2)
    plt.savefig('./figures_nf/hist_' + file_name + '_after.jpg', bbox_inches='tight', dpi=300)

    plt.figure()
    plt.imshow(z_latent.detach().cpu().numpy()[0,0,...], cmap='gray')
    plt.axis ('off')
    plt.savefig('./figures_nf/z_' + file_name + '_before.jpg', bbox_inches='tight', dpi=300)
    plt.figure()
    plt.imshow(z_out.detach().cpu().numpy()[0,0,...], cmap='gray')
    plt.axis ('off')
    plt.savefig('./figures_nf/z_' + file_name + '_after.jpg', bbox_inches='tight', dpi=300)

    plt.close('all')
    print(f'z before norm = {z_latent.reshape(-1).norm()}')
    print(f'z after norm = {z_out.reshape(-1).norm()}')

# heavy-tailed
print ('heavy tail')
np.random.seed(seed=1)
z = torch.randn(1, 3, nh, nw).to(torch.device('cuda'))
z = z * torch.exp (0.5 * 0.5 * z**2)
z = spherical(z)
exp_map(z*0.7, 'ht', gtrans=True, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=1.0)
exp_map(z, 'ht_lmbt', gtrans=True, if_ica=False, if_yj=False, if_lambt=True, if_standard=True, temp=0.7)
exp_map (z*0.7, 'ht_all0.7', lpsize=lpsize, gtrans=True, if_ica=True, if_yj=True, if_lambt=True, if_standard=True, temp=0.7)

# skewed
print('skewed' )
np.random.seed(seed=2)
print(stats.loggamma.stats(1, moments='mvsk'))
z = torch.from_numpy((stats.loggamma.rvs(1, size=3*nh*nw)).reshape(1, 3, nh, nw)).type(torch.float32).to(torch.device('cuda'))
z = spherical(z)
exp_map(z*0.7, 'sk0.7', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=1.0)
exp_map(z*0.7, 'sk_yj', gtrans=True, if_ica=False, if_yj=True, if_lambt=False, if_standard=True, temp=0.7)
exp_map(z*0.1, 'sk0.1', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=1.0)
exp_map(z*0.3, 'sk0.3', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=1.0)
exp_map(z*1.0, 'sk1.0', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=1.0)
exp_map(z*2.0, 'sk2.0', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=1.0)
exp_map(z*0.7, 'sk_all0.7', lpsize=lpsize, gtrans=True, if_ica=True, if_yj=True, if_lambt=True, if_standard=True, temp=0.7)

# sin-cos
x_sin = torch.sin(2*np.pi/nw * torch.arange(nw))
y_cos = torch.cos(2*np.pi/nh * torch.arange(nh))
z = torch.stack([torch.outer(x_sin, y_cos), torch.outer(x_sin, y_cos), torch.outer(x_sin, y_cos)], dim=0).unsqueeze(0).to(device)
noise_seed = 0
rs= np.random.RandomState(noise_seed)
z = z + torch.from_numpy(rs.randn(*z.shape)).to(device)*0.5
z = spherical(z) * 0.7
exp_map(z, 'sin', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=1.0)
exp_map(z, 'sin_gau_wht', lpsize=lpsize, gtrans=True, if_ica=True, if_yj=True, if_lambt=True, if_standard=True, ica_only_whiten=True, temp=0.7)
exp_map(z, 'sin_gau', lpsize=lpsize, gtrans=True, if_ica=True, if_yj=True, if_lambt=True, if_standard=True, temp=0.7)

# Gaussian vectors of different norms
noise_seed = 0
rs= np.random.RandomState(noise_seed)
z = torch.from_numpy(rs.randn(*z.shape)).to(device)
exp_map(z, 'gau0.7', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=0.7)
exp_map(z, 'gau0.1', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=0.1)
exp_map(z, 'gau1.0', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=1.0)
exp_map(z, 'gau2.0', gtrans=False, if_ica=False, if_yj=False, if_lambt=False, if_standard=False, temp=2.0)