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
import sys, os, copy, json
sys.path.append('../../../')
sys.path.append('../../../stylegan2-ada-pytorch')
from dgminv.optimize.obj_wrapper import PyTorchObjective
from dgminv.optimize.inversion_solver import InvSolver
from dgminv.utils.common import add_noise
from dgminv.networks.models_cond_glow import GlowInv
# using the first parameterization: extract patches from latent tensor before squeezing
from dgminv.networks.glow_wrapper1 import GlowWrap
from dgminv.ops.cs_models import GaussianSensor
from dgminv.ops.gaussian_op import GaussianSmoothing
from dgminv.utils.skewness import skew
from dgminv.utils.kurtosis import kurtosis
from dgminv.utils.img_metrics import psnr, ssim
import lpips
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--n_bits', default=5, type=int)
    parser.add_argument('--n_block', default=4, type=int)
    parser.add_argument('--n_flow', default=32, type=int)
    parser.add_argument('--no_lu', default=False, type=bool)
    parser.add_argument('--no_affine', default=True, type=bool)
    parser.add_argument('--no_cond', default=True, type=bool)
    parser.add_argument('--targets_dir', default='', type=str)
    parser.add_argument('--results_dir', default='', type=str)
    parser.add_argument('--network_weights', default='', type=str)
    parser.add_argument('--img_name', default='', type=str)
    parser.add_argument('--lpsize', default=8, type=int)
    parser.add_argument('--gtrans', default=0, type=int)
    parser.add_argument('--ortho', default=0, type=int)
    parser.add_argument('--subsample_ratio', default=1./20., type=float)
    parser.add_argument('--snr_noise', default=20.0, type=float)
    parser.add_argument('--smth_sigma', default=3.0, type=float)
    parser.add_argument('--noise_std', default=50.0, type=float)
    parser.add_argument('--temp', default=0.7, type=float)
    parser.add_argument('--sensor_type', choices=('GaussianCS', 'Smoothing'), default='Smoothing', type=str)
    parser.add_argument('--lseed', default=0, type=int)
    parser.add_argument('--smart_init', action='store_true')
    parser.add_argument('--ab_if_yj', default=1, type=int)
    parser.add_argument('--ab_if_lambt', default=1, type=int)
    parser.add_argument('--ab_if_ica', default=1, type=int)
    parser.add_argument('--ab_ica_only_whiten', default=0, type=int)
    parser.add_argument('--save_binary', action='store_true', help='save intermediate binary images')
    parser.add_argument('--cache_dir', default='./smart_init_cache', type=str)
    parser.add_argument('--if_spherical', default=0, type=int)
    return vars(parser.parse_args())

opt = get_args()

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
opt['device'] = device

snr_noise = opt['snr_noise']
noise_std = opt['noise_std']
subsmpratio = opt['subsample_ratio']
smth_sigma = opt['smth_sigma']
sensor_type = opt['sensor_type']
gtrans = bool(opt['gtrans'])
ortho = bool(opt['ortho'])
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
# For images, the value range is normalized to [0,1]
# img_true = mpimg.imread(os.path.join(img_addr, opt['img_name']+'.jpg')).astype(np.float32).transpose(2,0,1)/255.0
img_true = mpimg.imread(os.path.join(img_addr, opt['img_name']+'.jpg')).astype(np.float32).transpose(2,0,1)
assert opt['in_channel'] == img_true.shape[0]
nc, nh, nw = img_true.shape
n_pixel_img = np.sum(img_true.shape)
# th_img_true = torch.tensor(img_true).unsqueeze(0).to(device).clamp(0.0, 1.0)
th_img_true = torch.tensor(img_true).unsqueeze(0).to(device).clamp(0, 255)
print(f'th_img_true shape = {th_img_true.shape}')

if opt['sensor_type'] == 'GaussianCS':
    fwd_op = GaussianSensor(th_img_true.shape, subsample_ratio=subsmpratio)
elif opt['sensor_type'] == 'Smoothing':
    fwd_op = GaussianSmoothing(opt['in_channel'], smth_sigma).to(device)
else:
    raise NotImplementedError

def plot_imgs(th_img, result_dir_name, img_name, figsize=(10,5)):
    plt.figure(figsize=(10,5))
    # plt.imshow(th_img[0,...].cpu().numpy().transpose(1,2,0))
    plt.imshow(th_img[0,...].cpu().numpy().transpose(1,2,0)/255)
    plt.axis('off')
    plt.savefig(os.path.join(result_dir_name, img_name), bbox_inches = 'tight')
    plt.close('all')

plot_imgs(th_img_true, result_dir_name, 'Img_true.jpg')

# generate data
with torch.no_grad():
    if sensor_type == 'GaussianCS':
        th_obs_data = add_noise(fwd_op(th_img_true), SNR=snr_noise, mode='gaussian', seed=1234)
    elif sensor_type == 'Smoothing':
        th_obs_data = add_noise(fwd_op(th_img_true), sigma=noise_std, mode='gaussian', seed=1234)

if sensor_type == 'Smoothing':
    plot_imgs(th_obs_data, result_dir_name, 'Img_smooth.jpg')

with torch.no_grad():
    sio.savemat(f'{result_dir_name}/Img_true.mat', \
            {'Img_true':th_img_true[0,...].cpu().numpy().transpose(1,2,0)})
    sio.savemat(f'{result_dir_name}/Obs_data.mat', \
            {'Obs_data':th_obs_data.cpu().numpy()})


# =================== inversion =========================
# initialize glow
state_dict = torch.load(opt['network_weights'], map_location=torch.device('cpu'))
n_bins = 2 ** opt['n_bits']
glow_inv = GlowInv(opt['in_channel'], opt['n_flow'], opt['n_block'], \
    affine=not opt['no_affine'], conv_lu=not opt['no_lu'], cond=not opt['no_cond'])
glow_inv.eval()
glow_inv.load_state_dict(state_dict, strict=True)
for name, param in glow_inv.named_parameters():
    param.requires_grad = False
glow_inv.to(device)
model_inv = GlowWrap(glow_inv, nh=nh, nw=nw, nc=nc, n_bits=opt['n_bits'], \
    lpsize=opt['lpsize'], lseed=opt['lseed'], temp=opt['temp'], gtrans=gtrans, ortho=ortho, if_ica=ab_if_ica, \
        ica_only_whiten=ab_ica_only_whiten, if_yj=ab_if_yj, if_lambt=ab_if_lambt, if_spherical=if_spherical)
model_inv.to(device)

lossFunc = lambda x, y: 0.5 * torch.norm(x[:] - y[:])**2

if opt['smart_init']:
    num_init = 100
    print('Use smart initialization')
    cache_dir = opt['cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    lpsize, gtrans, ortho= opt['lpsize'], opt['gtrans'], opt['ortho']
    ifica, ow, yj, lambt = opt['ab_if_ica'], opt['ab_ica_only_whiten'], opt['ab_if_yj'], opt['ab_if_lambt']
    sensor_type = opt['sensor_type']
    if sensor_type == 'GaussianCS':
        cache_fname = f'GaussianCS_sub{subsmpratio}_lps{lpsize}_' \
            + f'gtrans{gtrans}_ortho{ortho}_' \
            + f'ica{ifica}_ow{ow}_yj{yj}_lambt{lambt}.npy'
    elif sensor_type == 'Smoothing':
        cache_fname = f'Deblur_sub{subsmpratio}_lps{lpsize}_' \
            + f'gtrans{gtrans}_ortho{ortho}_' \
            + f'ica{ifica}_ow{ow}_yj{yj}_lambt{lambt}.npy'
    else:
        raise NotImplementedError
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
                model_inv.set_latent(seed_init)
                th_curr_img,_,_ = model_inv()
                th_calc_data = fwd_op(th_curr_img)
                loss_init.append(lossFunc(th_calc_data, th_obs_data).item())
                th_calc_data_all.append(th_calc_data)
            th_calc_data_all = torch.stack(th_calc_data_all, dim=0)
            if not os.path.exists(cache_full_addr):
                np.save(cache_full_addr, th_calc_data_all.cpu().numpy())
        
    loss_init = np.array(loss_init)
    print(f'loss_init = {loss_init}')
    best_init_idx = np.argmin(loss_init)
    print(f"Picking {best_init_idx}")
    model_inv.set_latent(best_init_idx)


with torch.no_grad():
    th_img_check,_,_ = model_inv()
    plot_imgs(th_img_check, result_dir_name, 'Img_check.jpg')

psnr_calc = lambda th_ImgProg : psnr(th_img_true, th_ImgProg)
ssim_calc = lambda th_ImgProg : ssim(th_img_true, th_ImgProg)
lpips_calc = lambda th_ImgProg : loss_fn_alex(th_img_true/255.0, th_ImgProg/255.0, normalize=True)

plot_img_func = lambda x, name : plot_imgs(x, result_dir_name, name)
range_convert_func = lambda x : x 

inv_solver = InvSolver(th_obs_data, fwd_op, model_inv, lpips_calc, psnr_calc, ssim_calc, \
    range_convert_func=range_convert_func, plot_img_func=plot_img_func, result_dir_name=result_dir_name, 
    save_binary=opt['save_binary'], is_glow=True)

inv_solver.fit_lbfgs()