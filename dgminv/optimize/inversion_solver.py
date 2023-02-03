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
import h5py
import sys, os, copy, json
sys.path.append('../../../')
sys.path.append('../../../stylegan2-ada-pytorch')
from dgminv.optimize.obj_wrapper import PyTorchObjective
from dgminv.optimize.langevin import Langevin
from dgminv.utils.common import torch_dtype as td
from dgminv.utils.common import noise_reg, add_noise
from dgminv.utils.skewness import skew
from dgminv.utils.kurtosis import kurtosis
from dgminv.utils.img_metrics import psnr, ssim
import lpips

def convert_func(x):
    return torch.clamp(x, 0, 1)
    # return torch.clamp((x + 1.0)/2.0, 0, 1)

class InvSolver:
    def __init__(self, \
        th_obs_data,                        # observed data
        fwd_op,                             # forward physics model
        model_inv,                          # model parameterization 
        lpips_calc,                         # function to compute lpips
        psnr_calc,                          # function to compute psnr
        ssim_calc,                          # function to cmpute ssim
        range_convert_func=convert_func,    # convert output from model_inv to physical ranges
        plot_img_func=None,                 # image plotting function during inversion
        result_dir_name='',                 # directory to store results: str
        save_binary=False,                  # flag to save intermediate binary results: bool
        is_glow=False,
        ext_loss_func=None
        ):
        self.th_obs_data = th_obs_data
        self.fwd_op = fwd_op
        self.model_inv = model_inv
        self.range_convert_func = range_convert_func
        self.plot_img_func = plot_img_func
        self.result_dir_name = result_dir_name
        self.save_binary = save_binary
        self.is_glow = is_glow
        self.lpips_calc, self.psnr_calc, self.ssim_calc = lpips_calc, psnr_calc, ssim_calc
        self._iter, self._loc_counter = 0, 0
        self.losses = []
        self.lossFunc = lambda x, y: 0.5 * torch.norm(x[:] - y[:])**2
        self.save_every = 10
        self.ext_loss_func = ext_loss_func
        # adam optimizer params: from the stylegan2 projector
        self.adam_init_lr = 0.1
        self.initial_learning_rate      = 0.1
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5

    def compLoss(self):
        if self.is_glow:
            th_curr_img,_,_ = self.model_inv()
        else:
            th_curr_img = self.model_inv()
        
        th_curr_img = self.range_convert_func(th_curr_img)
        th_calc_data = self.fwd_op(th_curr_img)
        
        if self.ext_loss_func:
            loss = self.ext_loss_func(th_calc_data, self.th_obs_data)
        else:
            loss = self.lossFunc(th_calc_data, self.th_obs_data)
        
        if self._iter == self._loc_counter + 1:
            self.losses.append(loss.item())
            with open(self.result_dir_name + '/losses.txt', 'a') as text_file:
                text_file.write("%d %s\n" % (self._iter, loss.item()))
            self._loc_counter += 1
        return loss

    def save_prog(self, x):
        # with open(self.result_dir_name + '/loss.txt', 'a') as text_file:
        #     text_file.write("%d %s\n" % (self._iter, self.opt_obj.f))

        prefix = ''
        if isinstance(x, str):
            prefix = x

        if self._iter % self.save_every ==0:
            with torch.no_grad():
                if self.is_glow:
                    th_ImgProg,_,_ = self.model_inv()
                else:
                    th_ImgProg = self.model_inv()

                th_ImgProg = self.range_convert_func(th_ImgProg)

            self.plot_img_func(th_ImgProg, prefix + 'Img_inv_' + str(self._iter) + '.jpg')
            ImgProg = th_ImgProg[0,...].detach().cpu().numpy()

            # save image vectors
            if self.save_binary:
                with torch.no_grad():
                    np.save(f'{self.result_dir_name}/{prefix}Img_inv_vec_{self._iter}.npy', \
                        ImgProg)

            # compute image metrics
            with torch.no_grad():
                img_psnr = self.psnr_calc(th_ImgProg)
                img_ssim = self.ssim_calc(th_ImgProg)
                img_lpips = self.lpips_calc(th_ImgProg)
            with open(self.result_dir_name + f'/{prefix}img_metrics.txt', 'a') as text_file:
                text_file.write("%d %s %s %s\n" % (self._iter, img_psnr.item(), img_ssim.item(), img_lpips.item()))

        self._iter = self._iter + 1
    
    def reset_counter(self):
        self._iter, self._loc_counter = 0, 0
        self.save_every = 10
    
    def save_final(self, prefix=''):
        losses = np.array(self.losses)
        losses = losses / losses[0]
        np.save(os.path.join(self.result_dir_name, prefix+'data_loss.npy'), losses)
        plt.figure()
        plt.semilogy(np.arange(1, losses.shape[0]+1).astype(int), losses)
        plt.xlabel('Iterations')
        plt.ylabel('Normalized Data Loss')
        plt.savefig(self.result_dir_name + f'/{prefix}data_loss.jpg', dpi=300, bbox_inches='tight')

        with torch.no_grad():
            if self.is_glow:
                th_ImgProg,_,_ = self.model_inv()
            else:
                th_ImgProg = self.model_inv()
            th_ImgProg = self.range_convert_func(th_ImgProg)
            if self.is_glow:
                th_latent = self.model_inv.get_latent()
                np.save(f'{self.result_dir_name}/{prefix}Latent_final.npy', th_latent.detach().cpu().numpy())
        ImgProg = th_ImgProg[0,...].detach().cpu().numpy()
        np.save(f'{self.result_dir_name}/{prefix}Img_inv.npy', ImgProg)


    def fit_lbfgs(self, maxiter=510):
        self.opt_obj = PyTorchObjective(self.compLoss, self.model_inv, retain_graph=False)
        optimize.minimize(self.opt_obj.fun, self.opt_obj.x0, method='L-BFGS-B', jac=self.opt_obj.jac, bounds=self.opt_obj.bounds, \
        tol=None, callback=self.save_prog, options={'disp': True, 'iprint': 101, \
        'gtol': 1e-12, 'maxiter': maxiter, 'ftol': 1e-12, 'maxcor': 30, 'maxfun': 15000})
        self.save_final()


    def fit_adam(self, maxiter=1000):
        param_list = []
        for name, param in self.model_inv.named_parameters():
            if param.requires_grad == True:
                print('Parameter: ', name)
                param_list.append(param)
        optimizer = torch.optim.Adam(param_list, betas=(0.9, 0.999), lr=self.initial_learning_rate)
        for step in range(maxiter):
            optimizer.zero_grad(set_to_none=True)
            loss = self.compLoss()
            loss.backward()
            optimizer.step()
            self.save_prog(0)
        self.save_final()
    
    def fit_langevin(self, nsteps=1000):
        self.save_every = 100
        param_list = []
        for name, param in self.model_inv.named_parameters():
            if param.requires_grad == True:
                print('Parameter: ', name)
                param_list.append(param)
        self.initial_learning_rate = 0.0005
        optimizer = Langevin(param_list, lr=self.initial_learning_rate)
        for step in range(nsteps):
            # Annealing schedule.
            # lr = self.initial_learning_rate * max(0.0, 1.0 - step/nsteps) ** 2
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss = self.compLoss()
            loss.backward()
            optimizer.step()
            self.save_prog('lgv-')
        self.save_final('lgv-')

    # from projector.py in stylegan2-ada-pytorch
    def fit_noisereg(self, num_steps = 1000):
        # Please don't call model_inv() before this function! This will disable the update in the noise part.

        device = self.model_inv.noise_patches.device
        w_avg_samples = 10000
        G = copy.deepcopy(self.model_inv.G).eval().requires_grad_(False)
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
        print(f'w_avg shape = {w_avg.shape}')
        print(f'w shape ={self.model_inv.w.shape[0]}')
        print(f'G.num_ws ={G.num_ws}')
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
        num_ws = self.model_inv.w.shape[0]

        # Setup noise inputs.
        noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
        
        # w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
        w_opt = torch.tensor(np.tile(w_avg, [1, num_ws, 1]), dtype=torch.float32, device=device, requires_grad=True)

        # w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=self.initial_learning_rate)

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        for step in range(num_steps):
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = w_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
            lr = self.initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            if num_ws > 1:
                ws = w_opt + w_noise
            elif num_ws == 1:
                ws = (w_opt + w_noise).repeat([1, G.num_ws, 1])
            # ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
            
            # compute loss
            th_curr_img = G.synthesis(ws, noise_mode='const')
            th_curr_img = self.range_convert_func(th_curr_img)
            th_calc_data = self.fwd_op(th_curr_img)
        
            loss0 = self.lossFunc(th_calc_data, self.th_obs_data)

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = torch.nn.functional.avg_pool2d(noise, kernel_size=2)
            
            loss = loss0 + reg_loss * self.regularize_noise_weight
            print(f'reg_loss = {reg_loss}')

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (step + 1) % self.save_every == 0:
                self.plot_img_func(th_curr_img.detach().clone(), 'Img_inv_' + str(step) + '.jpg')
                with torch.no_grad():
                    img_psnr = self.psnr_calc(th_curr_img)
                    img_ssim = self.ssim_calc(th_curr_img)
                    img_lpips = self.lpips_calc(th_curr_img)
                with open(self.result_dir_name + '/img_metrics.txt', 'a') as text_file:
                    text_file.write("%d %s %s %s\n" % (step, img_psnr.item(), img_ssim.item(), img_lpips.item()))

            # Save projected W for each optimization step.
            # w_out[step] = w_opt.detach()[0]

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        with torch.no_grad():
            th_curr_img = G.synthesis(ws, noise_mode='const')
            th_curr_img = self.range_convert_func(th_curr_img)
            img_inv = th_curr_img[0,...].detach().cpu().numpy()
            np.save(f'{self.result_dir_name}/Img_inv.npy', img_inv)