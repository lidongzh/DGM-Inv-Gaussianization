# Heavily inspired by https://github.com/rosinality/glow-pytorch
import torch
from torch import nn
from torch.nn import functional as F
import torch.fft
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from scipy import signal
import torch.utils.checkpoint as cpoint
import sys
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        batch, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet.repeat(batch)

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        batch, _, height, width = output.shape
        log_abs = -logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)
        # print(f'actnorm logdet = {logdet.mean()}')
        if self.logdet:
            return output / self.scale - self.loc, -logdet.repeat(batch) 


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )

class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_s.setflags(write=1)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)
        
        self.Mat = None

    def forward(self, input):
        batch, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        # here the logdet is a number, since for all input they are the same
        return out, logdet.repeat(batch)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output, reconstruct=False):
        batch, _, height, width = output.shape
        if not reconstruct:
            weight = self.calc_weight()
            self.Mat = weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        else:
            if self.Mat == None:
                weight = self.calc_weight()
                self.Mat = weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)

        logdet = -height * width * torch.sum(self.w_s)
        # print(f'1x1 logdet = {logdet.mean()}')

        return F.conv2d(output, self.Mat), -logdet.repeat(batch)


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


# class AffineCoupling(nn.Module):
#     def __init__(self, in_channel, filter_size=512, affine=True, onehot=1):
#         super().__init__()

#         self.affine = affine

#         self.net = nn.Sequential(
#             nn.Conv2d(in_channel // 2 + onehot, filter_size, 3, padding=1), # add one channel of label
#             nn.ReLU(inplace=True),
#             nn.Conv2d(filter_size, filter_size, 1),
#             nn.ReLU(inplace=True),
#             ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
#         )

#         self.net[0].weight.data.normal_(0, 0.05)
#         self.net[0].bias.data.zero_()

#         self.net[2].weight.data.normal_(0, 0.05)
#         self.net[2].bias.data.zero_()

#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, input, label):
#         in_a, in_b = input.chunk(2, 1)

#         if self.affine:
#             log_s, t = self.net(torch.cat([in_a,label], 1)).chunk(2, 1)
#             # s = torch.exp(log_s)
#             s = self.sigmoid(log_s + 2.0)
#             # out_a = s * in_a + t
#             out_b = (in_b + t) * s

#             # here the logdet should be an array of the size of batch number
#             logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

#         else:
#             net_out = self.net(torch.cat([in_a, label], 1))
#             out_b = in_b + net_out
#             logdet = None

#         return torch.cat([in_a, out_b], 1), logdet

#     def reverse(self, output, label):
#         out_a, out_b = output.chunk(2, 1)

#         if self.affine:
#             log_s, t = self.net(torch.cat([out_a, label], 1)).chunk(2, 1)
#             # s = torch.exp(log_s)
#             s = self.sigmoid(log_s + 2.0)
#             # in_a = (out_a - t) / s
#             in_b = out_b / s - t

#             logdet = torch.sum(torch.log(s).view(output.shape[0], -1), 1)

#         else:
#             net_out = self.net(torch.cat([out_a, label], 1))
#             in_b = out_b - net_out
#             logdet = None

#         return torch.cat([out_a, in_b], 1), logdet


class Flow(nn.Module):
    def __init__(self, in_channel, sp_dim=1, affine=True, conv_lu=True, onehot=1, cond=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)
        
        self.cond = cond

        # self.coupling = AffineCoupling(in_channel, affine=affine, onehot=onehot)
        self.coupling = AffineCouplingCHV(in_channel, affine=affine, sp_dim=sp_dim, onehot=onehot, cond=cond)

    def forward(self, input, label=None):
        out, logdet = self.actnorm(input)
        # why first invconv?
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out, label)
        # out, det2 = self.coupling(out, label)
        # out, det1 = self.invconv(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output, label=None, reconstruct=False):
        input, det2 = self.coupling.reverse(output, label)
        input, det1 = self.invconv.reverse(input, reconstruct)
        # input, det1 = self.invconv.reverse(output, reconstruct)
        # input, det2 = self.coupling.reverse(input, label)
        input, logdet = self.actnorm.reverse(input)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return input, logdet


class AffineCouplingCHV(nn.Module):
    def __init__(self, in_channel, filter_size=512, sp_dim=1, affine=True, onehot=1, cond=True):
        super().__init__()

        self.affine = affine

        self.sp_dim = sp_dim

        self.cond = cond

        if sp_dim == 1:
            if self.cond:
                self.net = nn.Sequential(
                    nn.Conv2d(in_channel // 2 + onehot, filter_size, 3, padding=1), # add one channel of label
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filter_size, filter_size, 1),
                    nn.ReLU(inplace=True),
                    ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
                )
            else:
                self.net = nn.Sequential(
                    nn.Conv2d(in_channel // 2, filter_size, 3, padding=1), # add one channel of label
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filter_size, filter_size, 1),
                    nn.ReLU(inplace=True),
                    ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
                )

        elif sp_dim == 2 or sp_dim == 3:
            if self.cond:
                self.net = nn.Sequential(
                    nn.Conv2d(in_channel + onehot, filter_size, 3, padding=1), # add one channel of label
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filter_size, filter_size, 1),
                    nn.ReLU(inplace=True),
                    ZeroConv2d(filter_size, 2 * in_channel if self.affine else in_channel),
                )
            else:
                self.net = nn.Sequential(
                    nn.Conv2d(in_channel, filter_size, 3, padding=1), # add one channel of label
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filter_size, filter_size, 1),
                    nn.ReLU(inplace=True),
                    ZeroConv2d(filter_size, 2 * in_channel if self.affine else in_channel),
                )
        else:
            raise ValueError('split dim error!')

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, label=None):
        in_a, in_b = input.chunk(2, self.sp_dim)
        if self.cond:
            if self.sp_dim == 2 or self.sp_dim == 3:
                label, _ = label.chunk(2, self.sp_dim)

        if self.affine:
            if self.cond:
                log_s, t = self.net(torch.cat([in_a,label], 1)).chunk(2, 1)
            else:
                log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = self.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            # here the logdet should be an array of the size of batch number
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            if self.cond:
                net_out = self.net(torch.cat([in_a, label], 1))
            else:
                net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], self.sp_dim), logdet

    def reverse(self, output, label=None):
        out_a, out_b = output.chunk(2, self.sp_dim)
        if self.cond:
            if self.sp_dim == 2 or self.sp_dim == 3:
                label, _ = label.chunk(2, self.sp_dim)

        if self.affine:
            if self.cond:
                log_s, t = self.net(torch.cat([out_a, label], 1)).chunk(2, 1)
            else:
                log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = self.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

            logdet = torch.sum(torch.log(s).view(output.shape[0], -1), 1)
            # print(f'affine logdet = {logdet.mean()}')

        else:
            if self.cond:
                net_out = self.net(torch.cat([out_a, label], 1))
            else:
                net_out = self.net(out_a)
            in_b = out_b - net_out
            logdet = None

        return torch.cat([out_a, in_b], self.sp_dim), logdet



def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)
    
def standard_gaussian_log_p(x):
    return -0.5 * log(2 * pi)- 0.5 * (x) ** 2

def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True, onehot=1, cond=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, sp_dim=1, affine=affine, conv_lu=conv_lu, onehot=onehot, cond=cond))

        self.split = split
        self.cond = cond

        if split:
            if cond:
                self.prior = ZeroConv2d(in_channel * 2 + onehot, in_channel * 4) # add 1 channel of label
            else:
                self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            if cond:
                self.prior = ZeroConv2d(onehot, in_channel * 8) # 1 channel of label
            else:
                self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input, label=None):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out, label)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            if self.cond:
                mean, log_sd = self.prior(torch.cat([out, label], 1)).chunk(2, 1)
            else:
                mean, log_sd = self.prior(out).chunk(2, 1)

            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = (z_new - mean) / torch.exp(log_sd)

            # z_new = (z_new - mean) / torch.exp(log_sd)
            # log_p = standard_gaussian_log_p(z_new)
            # log_p = log_p.view(b_size, -1).sum(1)
            # logdet = logdet - log_sd.view(b_size, -1).sum(1)

        else:
            if self.cond:
                mean, log_sd = self.prior(label).chunk(2, 1)
            else:
                zero = torch.zeros_like(out)
                mean, log_sd = self.prior(zero).chunk(2, 1)

            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
            z_new = (z_new - mean) / torch.exp(log_sd)

            # z_new = out
            # z_new = (z_new - mean) / torch.exp(log_sd)
            # log_p = standard_gaussian_log_p(z_new)
            # log_p = log_p.view(b_size, -1).sum(1)
            # logdet = logdet - log_sd.view(b_size, -1).sum(1)

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, label=None, reconstruct=False):
        input = output
        b_size, n_channel, height, width = input.shape

        if self.split:
            if self.cond:
                mean, log_sd = self.prior(torch.cat([input, label], 1)).chunk(2, 1)
            else:
                mean, log_sd = self.prior(input).chunk(2, 1)

            z = gaussian_sample(eps, mean, log_sd)
            input = torch.cat([output, z], 1)
            log_p = gaussian_log_p(z, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

            # log_p = standard_gaussian_log_p(eps)
            # log_p = log_p.view(b_size, -1).sum(1)
            # logdet = -log_sd.view(b_size, -1).sum(1)
            # z = gaussian_sample(eps, mean, log_sd) # not gaussian, but reuse the same function to unstd
            # input = torch.cat([output, z], 1)

        else:
            # zero = torch.zeros_like(input)
            # zero = F.pad(zero, [1, 1, 1, 1], value=1)
            if self.cond:
                mean, log_sd = self.prior(label).chunk(2, 1)
            else:
                zero = torch.zeros_like(input)
                mean, log_sd = self.prior(zero).chunk(2, 1)

            z = gaussian_sample(eps, mean, log_sd)
            log_p = gaussian_log_p(z, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

            # log_p = standard_gaussian_log_p(eps)
            # log_p = log_p.view(b_size, -1).sum(1)
            # logdet = -log_sd.view(b_size, -1).sum(1)
            # z = gaussian_sample(eps, mean, log_sd)

            input = z

        logdet = 0
        for flow in self.flows[::-1]:
            input, det = flow.reverse(input, label, reconstruct)
            logdet += det

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed, logdet, log_p


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True, onehot=1, cond=True):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.n_block = n_block
        self.n_flow = n_flow
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, onehot=onehot, cond=cond))
            n_channel *= 2
        # The last block does not split
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine, onehot=onehot, cond=cond))
        # print(self.blocks)
        self.cond = cond

        # self.flows = nn.ModuleList()
        # for i in range(n_flow):
        #     self.flows.append(Flow(in_channel, sp_dim=i%2+2, affine=affine, conv_lu=conv_lu, onehot=onehot))
    
    def forward(self, input, label=None):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        # for flow in self.flows:
        #     out, det = flow(out, label)
        #     logdet = logdet + det

        for block in self.blocks:
            if self.cond:
                label = F.interpolate(label, scale_factor=0.5, mode='nearest')
                out, det, log_p, z_new = block(out, label)
            else:
                out, det, log_p, z_new = block(out)


            # out, det, log_p, z_new = cpoint.checkpoint(block,(out))
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, label=None, reconstruct=False):
        log_p_sum = 0
        logdet = 0
        scale = 0.5**self.n_block
        for i, block in enumerate(self.blocks[::-1]):
            if self.cond:
                curr_label = F.interpolate(label, scale_factor=scale, mode='nearest')
            else:
                curr_label = None
            scale *= 2
            if i == 0:
                input, det, log_p = block.reverse(z_list[-1], z_list[-1], label=curr_label, reconstruct=reconstruct)
            else:
                input, det, log_p = block.reverse(input, z_list[-(i + 1)], label=curr_label, reconstruct=reconstruct)

            logdet  = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p


        # for flow in self.flows[::-1]:
        #     input, det = flow.reverse(input, label, reconstruct)
        #     logdet += det
        
        return input, logdet, log_p_sum
    


class GlowInv(Glow):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True, onehot=1, cond=True):
        super(GlowInv, self).__init__(in_channel, n_flow, n_block, affine=affine, conv_lu=conv_lu, \
            onehot=onehot, cond=cond)
        # print(self.blocks)

    def squeeze(self, input):
        '''
        Convert a Tensor [B,C,H,W] into a list of squeezed tensors
        '''
        z_list = []
        z_in = input
        for i in range(self.n_block):
            b_size, n_channel, height, width = z_in.shape
            squeezed = z_in.view(b_size, n_channel, height // 2, 2, width // 2, 2)
            squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
            if i < self.n_block-1:
                z_in, z_new = out.chunk(2,1)
                z_list.append(z_new)
            else:
                z_list.append(out)
        return z_list
    
    def unsqueeze(self, z_list):
        b_size, n_channel, height, width = z_list[-1].shape

        unsqueezed = z_list[self.n_block-1]
        unsqueezed = unsqueezed.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )
        # print('last z:', unsqueezed.shape)
        n_channel, height, width = n_channel//4, height*2, width*2
        for i in range(self.n_block-2, -1, -1):
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

    def patches2z_list(self, input, label=None):
        '''
        To convert initial model patches to a latent space list 
        using pretrained glow network
        '''
        out = input
        z_outs = []

        for block in self.blocks:
            if self.cond:
                label = F.interpolate(label, scale_factor=0.5, mode='nearest')
            out, _, _, z_new = block(out, label)
            z_outs.append(z_new)

        return z_outs

    def forward(self, patchesLatent, label=None):
        # map from latent space vectors to a latent space list
        z_list = self.squeeze(patchesLatent) 
        # print([z_list[i].size() for i in range(len(z_list))])
        patches, logdet, log_p_sum = self.reverse(z_list, label, reconstruct=True)
        return patches, logdet, log_p_sum
