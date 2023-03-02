# Differentiable Gaussianization Layers for Inverse Problems Regularized by Deep Generative Models

This repository contains the implementation for 

[**Differentiable Gaussianization Layers for Inverse Problems Regularized by Deep Generative Models**](https://openreview.net/forum?id=OXP9Ns0gnIq) [**ICLR 2023**].

## Abstract
> Deep generative models such as GANs, normalizing flows, and diffusion models are powerful regularizers for inverse problems. They exhibit great potential for helping reduce ill-posedness and attain high-quality results. However, the latent tensors of such deep generative models can fall out of the desired high-dimensional standard Gaussian distribution during inversion, particularly in the presence of data noise and inaccurate forward models, leading to low-fidelity solutions. To address this issue, we propose to reparameterize and Gaussianize the latent tensors using novel differentiable data-dependent layers wherein custom operators are defined by solving optimization problems. These proposed layers constrain inverse problems to obtain high-fidelity in-distribution solutions. We validate our technique on three inversion tasks: compressive-sensing MRI, image deblurring, and eikonal tomography (a nonlinear PDE-constrained inverse problem) using two representative deep generative models: StyleGAN2 and Glow. Our approach achieves state-of-the-art performance in terms of accuracy and consistency.


## Usage
### Step 1:
1. Make sure that the Python version is 3.8 to reproduce the results and avoid package conflicts.
2. Make sure to turn on the `--recursive` flag when cloning the repo, so that the `stylegan2-ada-pytorch` submodule is also cloned.
3. (Optional) Create a python virtual environment:
    ```bash
    python3 -m venv myenv 
    source myenv/bin/activate
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2:
Download files:
1. Download `Eigen-3.3.7` and unpack it in `dgminv/lib`:
    ```bash
    mkdir dgminv/lib
    wget -O dgminv/lib/eigen-3.3.7.zip https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip
    unzip dgminv/lib/eigen-3.3.7.zip -d dgminv/lib/
    ```

2. Download the masks for MRI compressive sensing from: https://github.com/comp-imaging-sci/pic-recon/tree/main/pic_recon/masks_mri and put them under `tasks/stylegan2/inversion/masks_mri`

3. Download the weights from https://databank.illinois.edu/datasets/IDB-4499850 and put them under `tasks/stylegan2/inversion/weights`:
    ```bash
    mkdir tasks/stylegan2/inversion/weights
    wget -O tasks/stylegan2/inversion/weights/stylegan2-CompMRIT1T2-config-f.pkl https://databank.illinois.edu/datafiles/ln6ug/download
    ```

### Step 3:
Go into `tasks/stylegan2/inversion/`. Generate ground-truth images by runing 
```bash
python generate_mri_examples.py
```

### Step 4:
Run the following command to perform compressive sensing MRI inversion

```bash
python ./main_cs_sg2.py                 \
    --style_psize 32                    \
    --noise_psize 8                     \
    --targets_dir './targets'           \
    --results_dir './results'           \
    --cache_dir './cache'               \
    --snr_noise 20.0                    \
    --img_name 'mri_106'                  \
    --network_pkl './weights/stylegan2-CompMRIT1T2-config-f.pkl'        \
    --latent_mode 'mode-z+'             \
    --noise_update 1                    \
    --gtrans_noise 1                    \
    --gtrans_style 1                    \
    --ortho_noise 0                     \
    --ortho_style 0                     \
    --sensor_type MRI                   \
    --mri_mask './masks_mri/mask_rand_8x.npy' \
    --ab_if_ica 1                       \
    --ab_ica_only_whiten 0              \
    --ab_if_yj 1                        \
    --ab_if_lambt 0                     \
    --style_seed 2                      \
    --noise_seed 2
```
Meaning of parameters:
<br />
`style_psize`: patch size for style vectors                               <br /> 
`noise_psize`: patch size for noise vectors                               <br />
`targets_dir`: directory that contains ground-truth images                               <br />
`results_dir`: directory for results                               <br />
`cache_dir`: directory to contain cached data if using `--smart_init`                                 <br />
`snr_noise`: signal-to-noise ratio of data with added noise                                 <br />
`img_name`: name of image for inversion (.npy or .jpg)                                  <br />
`network_pkl`: address of network weights                               <br />
`latent_mode`: inversion mode: `mode-z+`, `mode-z`, `mode-w`                               <br />
`noise_update`: if updating noise vectors or not                              <br />
`gtrans_noise`: if using G layers on noise vectors or not                              <br />
`gtrans_style`: if using G layers on style vectors or not                              <br />
`ortho_noise`: if using orthogonal re-parameterization on noise vectors or not                               <br />
`ortho_style`: if using orthogonal re-parameterization on noise vectors or not                               <br />
`sensor_type`: type of compressive sensing operators: `MRI` or `Gaussian`                               <br />
`mri_mask`:  mask for compressive sensing MRI                          <br />
`ab_if_ica`: if turn on the ICA layer or not                                 <br />
`ab_ica_only_whiten`: if only use the whitening layer in the ICA layer                        <br />
`ab_if_yj`: if using the Yeo-Johnson layer or not                                <br />
`ab_if_lambt`: if using the Lambert F_X layer or not                               <br />
`style_seed`: seed for initializing style vectors                                <br />
`noise_seed`: seed for initializing noise vectors                                <br />

Similarly, run the following command to perform eikonal tomography
```bash
python ./main_tomo_sg2.py                 \
    --style_psize 32                    \
    --noise_psize 8                     \
    --targets_dir './targets'           \
    --results_dir './results'           \
    --cache_dir './cache'               \
    --img_name 'mri_104'                  \
    --network_pkl './weights/stylegan2-CompMRIT1T2-config-f.pkl'        \
    --latent_mode 'mode-z+'             \
    --noise_std 0.001                   \
    --noise_update 1                    \
    --gtrans_noise 1                    \
    --gtrans_style 1                    \
    --ortho_noise 0                     \
    --ortho_style 0                     \
    --src_stride 36                     \
    --ab_if_ica 1                       \
    --ab_ica_only_whiten 0              \
    --ab_if_yj 1                        \
    --ab_if_lambt 1                     \
    --style_seed 2                      \
    --noise_seed 2
```
### Comments
We loop through `style_seed=noise_seed=0,1,2` for each of the 100 or 25 images (see [More Details](More_details.md)), and report the best image / calculate metrics using the best score among the three runs. The [More Details](More_details.md) file also contains the images indices for the inversion examples shown in the paper.


## Training of Glow
Go into `tasks/glow/training`. Run
```python
python train.py PATH_TO_IMAGES
```
As mentioned by the author of the trainer, the training code uses ImageFolder from torchvision, so the images should be stored in directories structured like
> PATH_TO_IMAGES/class1 <br/>
> PATH_TO_IMAGES/class2 <br/>
> ...

One can simply put images in one class. For more details of training and how to prepare the CelebA-HQ dataset, please refer to Appendix F of the paper.


## Code structure
The structure of the repository is as follows:

```bash
dgminv # directory of the main source code
│
├── lib # download and unpack eigen-3.3.7 here
│
├── networks # contains networks and wrappers for inversion
│   ├── glow_wrapper1.py # wrapper for glow with the first parameterization
│   ├── glow_wrapper2.py # wrapper for glow with the second parameterization
│   ├── models_cond_glow.py # Glow network: inspired by https://github.com/rosinality/glow-pytorch (MIT license). One can also find a training script in this repository. We added conditioning functionality and wrapper for inversion. 
│   └── sgan_wrapper.py # wrapper for the generator of StyleGAN2
│
├── ops # operators
│   ├── eikonal # the Eikonal solver
│   ├── cs_models.py # forward models for compressive sensing
│   ├── gaussian_op.py # Gaussian smoothing operators
│   ├── ica.py # the ICA layer
│   ├── lambert_transform.py # the Lambert $W \times F_X$ layer
│   ├── lambertw_custom.py # custom implementatio of the Lambert W function
│   ├── ortho_trans.py # orthogonal re-parameterization
│   └── yeojohnson.py # the Yeo-Johnson layer
│       
├── optimize # optimizers and wrappers
│   ├── brent_optimizer.py # the Brent optimizer
│   ├── inversion_solver.py # wrapper of inversion solvers: L-BFGS-B, ADAM, Langevin dynamics, and noise regularizaiton
│   ├── langevin.py # Langevin dynamics
│   └── obj_wrapper.py # wrapper of Torch modules for SciPy optimizers
│
└── utils # auxiliary functions
    ├── common.py # helper functions
    ├── grad_check.py # class to check accuracy of gradients of custom ops
    ├── img_metrics.py # SSIM & PSNR
    ├── kurtosis.py # differentiable kurtosis function
    ├── lambertw.py # differentiable Lambert W function
    ├── lambertw2.py # another implementation of the Lambert W function
    ├── moments.py # compute moments
    └── skewness.py # compute skewness

stylegan2-ada-pytorch # The official implementation of StyleGAN2-ada: https://github.com/NVlabs/stylegan2-ada-pytorch.git

tasks # scripts for inversion/training/experiments
│
├── glow
│    ├── training/train.py # script to train the Glow model
│    └── inversion # inversion using Glow
│          ├── main_glow_inv_p1.py # inversion script with parameterization 1 for Glow
│          └── main_glow_inv_p2.py # inversion script with parameterization 2 for Glow
│
│── stylegan2/inversion # inversion using StyleGAN2
│    ├── generate_mri_examples.py # generate ground-truth examples for inversion
│    ├── main_cs_sg2.py # inversion script for compressive sensing using StyleGAN2
│    └── main_tomo_sg2.py # inversion script for Eikonal tomography using StyleGAN2
│
└── experiments
     ├── glow_p1_exp.py # script to generate the motivating examples for Glow
     └── stable_diffusion_exp.py # script to generate the motivating examples for Stable Diffusion
```


## License
This code is being shared under the [MIT license](LICENSE).


## Citation
```
@inproceedings{
li2023differentiable,
title={Differentiable Gaussianization Layers for Inverse Problems Regularized by Deep Generative Models},
author={Dongzhuo Li},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=OXP9Ns0gnIq}
}
```