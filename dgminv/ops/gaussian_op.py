import math
import numpy as np
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Modified from https://github.com/elephant-track/elephant-server/blob/main/elephant-core/elephant/util/gaussian_smoothing.py
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, sigma, radius=None, dim=2):
        super(GaussianSmoothing, self).__init__()
        sigma = float(sigma)
        if radius == None:
            radius = int(4.0 * sigma + 0.5)
            print(f'radius = {radius}')

        sigma2 = sigma * sigma
        x = np.arange(-radius, radius+1)
        # print(f'x = {x}')
        kernel_1d = np.exp(-0.5 / sigma2 * x ** 2)[::-1]
        # print(kernel_1d.size)
        kernel_1d = kernel_1d / kernel_1d.sum()
        th_kernel_1d = torch.tensor(kernel_1d, dtype=torch.float32)
        kernel = torch.outer(th_kernel_1d, th_kernel_1d)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.pad = (kernel_1d.size - 1) // 2

    def forward(self, input):
        input = F.pad(input, pad=(self.pad,self.pad,self.pad, self.pad), mode='reflect')
        return self.conv(input, weight=self.weight, padding=0, groups=self.groups)



class GaussianSmoothingVertical(nn.Module):
    def __init__(self, channels, sigma, radius=None, dim=2):
        super().__init__()
        sigma = float(sigma)
        if radius == None:
            radius = int(4.0 * sigma + 0.5)
            print(f'radius = {radius}')

        sigma2 = sigma * sigma
        x = np.arange(-radius, radius+1)
        kernel_1d = np.exp(-0.5 / sigma2 * x ** 2)[::-1]
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = torch.tensor(kernel_1d, dtype=torch.float32)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, kernel.shape[0], 1)
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.pad = (kernel_1d.size - 1) // 2

    def forward(self, input):
        input = F.pad(input, pad=(0,0,self.pad, self.pad), mode='reflect')
        return self.conv(input, weight=self.weight, padding=0, groups=self.groups)


class GaussianSmoothingHorizontal(nn.Module):
    def __init__(self, channels, sigma, radius=None, dim=2):
        super().__init__()
        sigma = float(sigma)
        if radius == None:
            radius = int(4.0 * sigma + 0.5)

        sigma2 = sigma * sigma
        x = np.arange(-radius, radius+1)
        # print(f'x = {x}')
        kernel_1d = np.exp(-0.5 / sigma2 * x ** 2)[::-1]
        # print(kernel_1d.size)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = torch.tensor(kernel_1d, dtype=torch.float32)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, 1, kernel.shape[0])
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.pad = (kernel_1d.size - 1) // 2

    def forward(self, input):
        input = F.pad(input, pad=(self.pad, self.pad, 0, 0), mode='reflect')
        return self.conv(input, weight=self.weight, padding=0, groups=self.groups)