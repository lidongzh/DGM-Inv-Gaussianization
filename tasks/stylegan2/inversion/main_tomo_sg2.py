import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import numpy as np
import scipy.io as sio
from scipy import optimize, stats
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import h5py
import sys, os, copy, json, copy
sys.path.append('../../../')
sys.path.append('../../../stylegan2-ada-pytorch')
from dgminv.optimize.obj_wrapper import PyTorchObjective
from dgminv.optimize.inversion_solver import InvSolver
from dgminv.utils.common import torch_dtype as td
from dgminv.utils.common import add_noise, tv
from dgminv.networks.sgan_wrapper import StyleGAN2Wrap
from dgminv.ops.eikonal.eikonal_ops import eikonal
from dgminv.ops.gaussian_op import GaussianSmoothing
from dgminv.utils.skewness import skew
from dgminv.utils.kurtosis import kurtosis
from dgminv.utils.img_metrics import psnr, ssim
import lpips
import argparse

class BaseModelInv(nn.Module):
    def __init__(self, model):
        super().__init__()
        if isinstance(model, np.ndarray):
            th_model = torch.tensor(model).type(torch.float32).view(1,1,model.shape[0],model.shape[1])
        else:
            th_model = model

        self.model = nn.Parameter(2.0 * th_model - 1.0)
        self.Bounds = {}

    def forward(self):
        return self.model

def img_convert(x):
    return torch.clamp((x + 1.0)/2.0, 0, 1)

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--style_psize', default=8, type=int)
    parser.add_argument('--noise_psize', default=8, type=int)
    parser.add_argument('--style_seed', default=0, type=int)
    parser.add_argument('--noise_seed', default=0, type=int)
    parser.add_argument('--network_pkl', default='', type=str)
    parser.add_argument('--targets_dir', default='', type=str)
    parser.add_argument('--results_dir', default='', type=str)
    parser.add_argument('--img_name', default='', type=str)
    parser.add_argument('--latent_mode', default='', type=str)
    parser.add_argument('--noise_update', default=1, type=int)
    parser.add_argument('--gtrans_noise', default=0, type=int)
    parser.add_argument('--ortho_noise', default=0, type=int)
    parser.add_argument('--gtrans_style', default=0, type=int)
    parser.add_argument('--ortho_style', default=0, type=int)
    # parser.add_argument('--subsample_ratio', default=1./20., type=float)
    parser.add_argument('--noise_std', default=0.0, type=float)
    parser.add_argument('--src_stride', default=2, type=int)
    parser.add_argument('--smart_init', action='store_true')
    parser.add_argument('--ab_if_yj', default=1, type=int)
    parser.add_argument('--ab_if_lambt', default=1, type=int)
    parser.add_argument('--ab_if_ica', default=1, type=int)
    parser.add_argument('--ab_ica_only_whiten', default=0, type=int)
    parser.add_argument('--save_binary', action='store_true', help='save intermediate binary images')
    parser.add_argument('--cache_dir', default='./smart_init_cache', type=str)
    parser.add_argument('--tv_beta', default=None, type=float)
    parser.add_argument('--if_spherical', default=0, type=int)
    return vars(parser.parse_args())

opt = get_args()

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
opt['device'] = device

noise_std = opt['noise_std']
src_stride = opt['src_stride']
noise_update = bool(opt['noise_update'])
gtrans_noise = bool(opt['gtrans_noise'])
gtrans_style = bool(opt['gtrans_style'])
ortho_noise = bool(opt['ortho_noise'])
ortho_style = bool(opt['ortho_style'])
ab_if_ica = bool(opt['ab_if_ica'])
ab_ica_only_whiten = bool(opt['ab_ica_only_whiten'])
ab_if_yj = bool(opt['ab_if_yj'])
ab_if_lambt = bool(opt['ab_if_lambt'])
if_spherical = bool(opt['if_spherical'])

loss_fn_alex = lpips.LPIPS(net='alex').to(device)

result_dir_name = opt['results_dir']
print(f'result_dir_name = {result_dir_name}')
os.makedirs(result_dir_name, exist_ok=True)

img_addr = opt['targets_dir']
# The value range is [-1,1] -> [0,1]
img_true = np.load(os.path.join(img_addr, opt['img_name']+'.npy')).astype(np.float32)
img_true = np.clip((img_true + 1.0) / 2.0, 0.0, 1.0)
shapey, shapex = [1, 256, 256], [1, 256, 256]
img_true = img_true.reshape(shapex)

n_pixel_img = np.sum(img_true.shape)
th_img_true = torch.tensor(img_true).unsqueeze(0).to(device).clamp(0.0, 1.0)
print(f'th_img_true shape = {th_img_true.shape}')

def plot_imgs(th_img, result_dir_name, img_name, figsize=(10,5)):
    plt.figure(figsize=(10,5))
    plt.imshow(th_img[0,...].cpu().numpy().transpose(1,2,0), cmap='gray', clim=[0., 1.])
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(result_dir_name, img_name), bbox_inches = 'tight')
    plt.close('all')

plot_imgs(th_img_true, result_dir_name, 'Img_true.jpg')

# source receiver geometry
h = 0.001 # grid spacing
nz, nx = th_img_true.shape[2], th_img_true.shape[3]
src_mask = np.zeros((nz, nx))
src_mask[2:nz:src_stride,0] = 1
src_mask[0,2:nx:src_stride] = 1
src_mask[2:nz:src_stride,nx-1] = 1
src_mask[nz-1,2:nx:src_stride] = 1
ind = np.argwhere(src_mask==1)
z_src = ind[:,0]
x_src = ind[:,1]
rec_stride = 1
rec_mask = np.zeros((nz, nx))
rec_mask[0:nz:rec_stride,0] = 1
rec_mask[0,0:nx:rec_stride] = 1
rec_mask[0:nz:rec_stride,nx-1] = 1
rec_mask[nz-1,0:nx:rec_stride] = 1
rec_dict = dict()
for (z, x) in zip(z_src, x_src):
    rec_mask_curr = copy.deepcopy(rec_mask)
    if z == 0 or z == nz - 1:
        rec_mask_curr[z, :] = 0
    if x == 0 or x == nx - 1:
        rec_mask_curr[:, x] = 0
    ind = np.argwhere(rec_mask_curr==1)
    z_rec = ind[:,0]
    x_rec = ind[:,1]
    rec_dict[(z, x)] = [z_rec, x_rec]

# convert from [0,1] to slowness
def img2phys(latentImg):
  return 1./(latentImg*100 + 1500.0)

def fwd_op(th_img, if_data_noise=False):
    assert len(th_img.shape) == 4
    th_slowness = img2phys(th_img)
    th_data = []
    for (k, (z,x)) in enumerate(zip(z_src, x_src)):
        th_data.append(eikonal(th_slowness[0,0,...], z, x, h)[rec_dict[(z, x)]])
    if if_data_noise:
        for idata in range(len(th_data)):
            th_data[idata] = add_noise(th_data[idata], sigma=noise_std, mode='speckle', seed=1234)
    th_data = torch.stack(th_data, dim=0)
    return th_data

with torch.no_grad():
    th_obs_data = fwd_op(th_img_true, if_data_noise=True)

with torch.no_grad():
    np.save(f'{result_dir_name}/Img_true.npy', \
        th_img_true[0,...].cpu().numpy().transpose(1,2,0))
    np.save(f'{result_dir_name}/Obs_data.npy', \
        th_obs_data.cpu().numpy())



# =================== inversion =========================
# initialize StyleGAN2 wrapper

if opt['tv_beta'] == None:
    model_inv = StyleGAN2Wrap(opt['network_pkl'], style_psize=opt['style_psize'], noise_psize=opt['noise_psize'], \
        style_seed=opt['style_seed'], noise_seed=opt['noise_seed'], mode=opt['latent_mode'], \
            noise_update=noise_update, gtrans_noise=gtrans_noise, 
            gtrans_style=gtrans_style, ortho_noise=ortho_noise,\
            ortho_style=ortho_style, if_ica=ab_if_ica, \
            ica_only_whiten=ab_ica_only_whiten, if_yj=ab_if_yj, if_lambt=ab_if_lambt, if_spherical=if_spherical)
    model_inv.to(torch.device('cuda'))
    print('noise', opt['noise_update'])
    print('gtrans_style', opt['gtrans_style'])
    for name, param in model_inv.named_parameters():
        if param.requires_grad == True:
            print('Parameter: ', name)
    lossFunc = lambda x, y: 0.5 * torch.norm(x[:] - y[:])**2 * 1e10 # the original loss scale is too small (1e-10)
else:
    gs = GaussianSmoothing(1, 11).to(device)
    th_img_sm = gs(th_img_true)
    model_inv = BaseModelInv(th_img_sm)
    model_inv.to(device)
    low_bounds = np.ones(model_inv.model.shape)
    up_bounds = np.ones(model_inv.model.shape)
    low_bounds[:,0,:,:] = -1.0 
    up_bounds[:,0,:,:] = 1.0
    model_inv.Bounds['model'] = [low_bounds, up_bounds]
    lossFunc = lambda x, y: 0.5 * torch.norm(x[:] - y[:])**2 * 1e10 # the original loss scale is too small (1e-10)
    def lossFunc(x, y):
        loss1 = 0.5 * torch.norm(x[:] - y[:])**2 * 1e10
        th_tv_z, th_tv_x = tv(model_inv().view(1,1,nz,nx))
        loss_tv = torch.nn.functional.l1_loss(th_tv_z[0,0,:,:], torch.zeros_like(th_tv_z[0,0,:,:])) \
            + torch.nn.functional.l1_loss(th_tv_x[0,0,:,:], torch.zeros_like(th_tv_x[0,0,:,:])) 
        loss2 = opt['tv_beta'] * loss_tv 
        print(f'loss1 = {loss1.item()}, loss2 = {loss2.item()}')
        return loss1 + loss2


if opt['smart_init']:
    num_init = 100
    print('Use smart initialization')
    cache_dir = opt['cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    latent_mode, style_psize, noise_update, gtrans_style, gtrans_noise, ortho_style, ortho_noise = \
        opt['latent_mode'], opt['style_psize'], opt['noise_update'], opt['gtrans_style'], \
            opt['gtrans_noise'], opt['ortho_style'], opt['ortho_noise']
    ifica, ow, yj, lambt = opt['ab_if_ica'], opt['ab_ica_only_whiten'], opt['ab_if_yj'], opt['ab_if_lambt']
    cache_fname = f'Tomo_stride{src_stride}_{latent_mode}_stps{style_psize}_' \
        + f'noiseupd{noise_update}_gstyle{gtrans_style}_gnoise{gtrans_noise}_' \
        + f'orthostyle{ortho_style}_orthonoise{ortho_noise}_ica{ifica}_ow{ow}_yj{yj}_lambt{lambt}.npy'
    cache_full_addr = os.path.join(cache_dir, cache_fname)
    print(f'cache_full_addr = {cache_full_addr}')
    try:
        calc_data_all = np.load(cache_full_addr)
    except:
        calc_data_all = np.array([])
    
    loss_init = []
    with torch.no_grad():
        if calc_data_all.shape[0] == num_init:
            print('Using cached...')
            th_calc_data_all = torch.from_numpy(calc_data_all).to(device) 
            for seed_init in range(num_init):
                loss_init.append(lossFunc(th_calc_data_all[seed_init], th_obs_data).item())
        else:            
            th_calc_data_all = []
            for seed_init in range(num_init):
                model_inv.set_latent(seed_init, seed_init)
                th_curr_img = model_inv()
                th_curr_img = img_convert(th_curr_img)
                th_calc_data = fwd_op(th_curr_img)
                loss_init.append(lossFunc(th_calc_data, th_obs_data).item())
                th_calc_data_all.append(th_calc_data)
            th_calc_data_all = torch.stack(th_calc_data_all, dim=0)
            if not os.path.exists(cache_full_addr):
                print('Dump cached...')
                np.save(cache_full_addr, th_calc_data_all.cpu().numpy())
        
    loss_init = np.array(loss_init)
    print(f'loss_init = {loss_init}')
    best_init_idx = np.argmin(loss_init)
    print(f"Picking {best_init_idx}")
    model_inv.set_latent(best_init_idx, best_init_idx)


with torch.no_grad():
    th_img_check = model_inv()
    th_img_check = img_convert(th_img_check)
#   print(f'min = {th_img_check.min()}, max = {th_img_check.max()}')
    plot_imgs(th_img_check, result_dir_name, 'Img_check.jpg')
    th_calc_data = fwd_op(th_img_check)
    np.save(f'{result_dir_name}/check_data.npy', th_calc_data.cpu().numpy())

psnr_calc = lambda th_ImgProg : psnr(th_img_true*255., th_ImgProg*255.)
ssim_calc = lambda th_ImgProg : ssim(th_img_true*255., th_ImgProg*255.)
lpips_calc = lambda th_ImgProg : loss_fn_alex(th_img_true, th_ImgProg, normalize=True)

plot_img_func = lambda x, name : plot_imgs(x, result_dir_name, name)
range_convert_func = lambda x : img_convert(x)

inv_solver = InvSolver(th_obs_data, fwd_op, model_inv, lpips_calc, psnr_calc, ssim_calc, \
    range_convert_func=range_convert_func, plot_img_func=plot_img_func, result_dir_name=result_dir_name, 
    save_binary=opt['save_binary'], ext_loss_func=lossFunc)


inv_solver.fit_lbfgs()