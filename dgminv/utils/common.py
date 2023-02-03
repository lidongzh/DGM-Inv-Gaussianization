import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../../')
from time import time
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy import signal, interpolate
from scipy.signal import convolve2d, convolve
# from dgminv.models.Laplacian_pyramid import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import yeojohnson_normmax
from dgminv.optimize.brent_optimizer import brent
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch_dtype = torch.float32

def img_to_patches(ImgIn, winSize, stride, nc):
    '''
    Convert a tensor to patches
    The tensor has a shape of [1,C,H,W]
    '''
    if (stride > winSize):
        raise ValueError('stride should not be larger than winsize!')
    nz,nx = ImgIn.size(2), ImgIn.size(3)
    padZ = int(np.floor((nz-1)/stride)) * stride + (winSize-1) - nz + 1
    padX = int(np.floor((nx-1)/stride)) * stride + (winSize-1) - nx + 1
    # print(f'padZ = {padZ}, padX = {padX}')
    # ImgInPad = F.pad(ImgIn, pad=(0,padX,0,padZ), \
    #     mode='constant', value=0.0)
    ImgInPad = F.pad(ImgIn, pad=(0,padX,0,padZ), \
        mode='circular')
    vecImgIn = F.unfold(ImgInPad, winSize, stride=stride)
    patches = torch.transpose(vecImgIn,1,2).view(vecImgIn.size(2), \
        nc, winSize, winSize)
    return patches

def patches_to_img(patches, nz, nx, winSize, stride, nc):
    '''
    Convert patches back to a tensor
    The patches has a shape of [B,C,winSize,winSize]
    This operator is the adjoint not the inverse to img_to_patches
    '''
    padZ = int(np.floor((nz-1)/stride)) * stride + (winSize-1) - nz + 1
    padX = int(np.floor((nx-1)/stride)) * stride + (winSize-1) - nx + 1
    vecImg = patches.reshape(1, patches.size(0), nc*patches.size(2)*patches.size(3)).transpose(1,2)

    ImgTemp = F.fold(vecImg, (nz+padZ, nx+padX), winSize, stride=stride)
    ImgOut = ImgTemp[:,:,:nz,:nx]
    return ImgOut

# shift a fourth order tensor down in the 2nd dimension (starting from 0)
def roll_z(x, n=1):  
    return torch.cat((x[:,:,-n:,:], x[:,:,:-n,:]), dim=2)
# keep the first row the same
def roll_z1(x):  
    return torch.cat((x[:,:,0,:].unsqueeze(2), x[:,:,:-1,:]), dim=2)

def roll_x1(x, n=1):  
    return torch.cat((x[:,:,:,0].unsqueeze(3), x[:,:,:,:-1]), dim=3)

def tv(img):
    img_z = roll_z1(img)
    img_x = roll_x1(img)
    return (img-img_z), (img-img_x)


def unsqueeze(z_list, nblock):
    b_size, n_channel, height, width = z_list[-1].shape

    unsqueezed = z_list[nblock-1]
    unsqueezed = unsqueezed.view(b_size, n_channel // 4, 2, 2, height, width)
    unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
    unsqueezed = unsqueezed.contiguous().view(
        b_size, n_channel // 4, height * 2, width * 2
    )
    # print('last z:', unsqueezed.shape)
    n_channel, height, width = n_channel//4, height*2, width*2
    for i in range(nblock-2, -1, -1):
        n_channel= n_channel//2
        # print('n_channel={}, height={}, width={}'.format(n_channel, height, width))
        unsqueezed = torch.cat([unsqueezed, z_list[i]], 1)
        unsqueezed = unsqueezed.view(b_size, n_channel, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        height, width = height*2, width*2
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel, height, width
        )
    return unsqueezed

def squeeze(input, n_block):
    '''
    Convert a Tensor [B,C,H,W] into a list of squeezed tensors
    '''
    z_list = []
    z_in = input
    for i in range(n_block):
        b_size, n_channel, height, width = z_in.shape
        squeezed = z_in.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        if i < n_block-1:
            z_in, z_new = out.chunk(2,1)
            z_list.append(z_new)
        else:
            z_list.append(out)
    return z_list

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes

def img_scale(ImgIn, low_bounds, up_bounds, normalize=True):
    '''
    map an image to the range of [0, 1]
    '''
    ImgOut = torch.zeros_like(ImgIn)
    for i in range(len(low_bounds)):
        if normalize:
            ImgOut[:,i,:,:] = (ImgIn[:,i,:,:] - low_bounds[i])/ (up_bounds[i] - low_bounds[i])
        else:
            ImgOut[:,i,:,:] = (up_bounds[i] - low_bounds[i]) * (ImgIn[:,i,:,:]) + low_bounds[i]

    return ImgOut

def scale_down_bits(ImgIn, low_bounds, up_bounds, n_bits):
    # scale to the range to images
    x = img_scale(ImgIn, low_bounds, up_bounds) * 255
    n_bins = 2 ** n_bits
    x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    return x

def scale_up_bits(ImgOut, low_bounds, up_bounds, n_bits):
    n_bins = 2 ** n_bits
    # x = torch.floor((ImgOut + 0.5) * n_bins) * (1.0 / n_bins)
    # x = torch.floor((ImgOut + 0.5) * n_bins) * 2**(8-n_bits)/255
    x = (ImgOut + 0.5) * n_bins * 2**(8-n_bits)/255
    x = torch.clamp(x, 0, 1)
    x = img_scale(x, low_bounds, up_bounds, normalize=False)
    return x

def perm_tensor_batch(x):
    x2 = x.reshape(x.shape[0], -1)
    out = torch.empty_like(x2)
    for i in range(x2.shape[0]):
        out[i,:] = x2[i,torch.randperm(x2.shape[1])]
    out = out.reshape(x.shape).contiguous()
    return out

def roll_tensor_batch(x, h=8, w=8, mode='forward'):
    # h,w = x.shape[2], x.shape[3]
    if mode == 'forward':
        out = torch.roll(x, shifts=(h//2, w//2), dims=(2,3))
    else:
        out = torch.roll(x, shifts=(-h//2, -w//2), dims=(2,3))
    return out


# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/projector.py
def noise_reg(z):
    ''' compute noise regularization term according to the StyleGAN2 paper
        z: [1, C, H, W]
    '''
    noise_reg_loss = 0.0
    for i in range(z.shape[1]):
        noise = z[:,i,:,:].unsqueeze(1)
        while True:
            noise_reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
            noise_reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return noise_reg_loss

def add_noise(x, SNR=None, sigma=None, mode='gaussian', seed=None):
    """ Adds gaussian noise of a given SNR to a signal
    """

    rnd = np.random if (seed == None) else np.random.RandomState(seed)

    assert(SNR != None or sigma != None)
    if sigma == None:
        p_signal = torch.norm(x)**2
        snr_inv = 10**(-0.1*SNR)

        p_noise = p_signal * snr_inv
        sigma = torch.sqrt(p_noise/np.prod(x.shape) ).type(x.dtype).to(x.device)

    if mode=='gaussian':
        x_noisy = x + sigma * torch.from_numpy(rnd.randn(*(x.shape))).type(x.dtype).to(x.device)
    elif mode=='salt_pepper':
        x_noisy = x + sigma * torch.from_numpy(abs(rnd.randn(*(x.shape)))).type(x.dtype).to(x.device)
    elif mode=='complex':
        x_noisy = x + sigma/np.sqrt(2) * torch.from_numpy(rnd.randn(*(x.shape)) \
            + 1.j*rnd.randn(*(x.shape))).type(x.dtype).to(x.device)
    elif mode=='speckle' and SNR == None:
        x_noisy = x + x * sigma * torch.from_numpy(rnd.randn(*(x.shape))).type(x.dtype).to(x.device)
    else:
        raise ValueError("Enter a suitable mode")

    return x_noisy

def timeit(func):
    # https://www.geeksforgeeks.org/timing-functions-with-decorators-python/
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func
