import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import numpy as np
import sys, os, copy, json
sys.path.append('../../../')
sys.path.append('../../../stylegan2-ada-pytorch')
from dgminv.networks.sgan_wrapper import StyleGAN2Wrap

network_pkl = './weights/stylegan2-CompMRIT1T2-config-f.pkl'
save_path = './targets'

os.makedirs(os.path.join(save_path, 'figures'), exist_ok=True)
# Selected images that were plausible by visual inspection
img_list = [0, 2, 9, 12, 13, 14, 15, 17, 20, 21, 23, 25, 26, 27,\
    29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 45, 46,\
    47, 48, 49, 50, 51, 53, 55, 57, 58, 60, 61, 64, 65, 67, 68,\
    69, 70, 71, 72, 73, 74, 75, 76, 79, 82, 83, 84, 85, 86, 87,\
    88, 89, 91, 92, 95, 96, 97, 98, 100, 101, 103, 104, 105, 106,\
    108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119,\
    122, 123, 124, 126, 127, 129, 130, 131, 132, 133, 135, \
    136, 137, 138, 140, 142]
for i in img_list:
    print(f'img {i}')
    style_seed = 100000 + i
    noise_seed = 200000 + i
    sg2 = StyleGAN2Wrap(network_pkl, style_seed=style_seed, noise_seed=noise_seed, mode='mode-z')
    with torch.no_grad():
        img = sg2()
    print(sg2.style_seed)
    plt.figure()
    img_np = img.detach().cpu().numpy()[0,0,...].clip(-1, 1)
    plt.imshow(img_np, cmap='gray', clim=[-1.0, 1.0])
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f'figures/mri_{i}.jpg'))
    plt.close()
    np.save(os.path.join(save_path, f'mri_{i}.npy'), img_np)
    
