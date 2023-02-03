# Based on https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb
import numpy as np
import torch
from scipy import stats
import sys, os
from PIL import Image
sys.path.append('../../')
from dgminv.utils.common import img_to_patches, patches_to_img
from dgminv.ops.yeojohnson import yeojohnson_transform as yj_transform
from dgminv.ops.lambert_transform import lambert_transform
from dgminv.ops.ica import FastIca
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use ('Agg')
plt.rcParams.update({'font.size': 40})
matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)


def normalize(x):
    return (x - torch.mean(x)) / (torch.std(x) + 1e-6)
def spherical(x):
  return x / torch.norm(x) * np.sqrt(np.prod(x.shape))

def glayers(z, lpsize=4, if_ica=True, 
    ica_only_whiten=False, 
    if_yj=True, 
    if_lambt=True,
    if_standard=True,
    if_spherical=False):
    # input: z: Tensor, latent tensor, [1, C, H, W]
    nc, nh, nw = z.shape[-3], z.shape[-2], z.shape[-1]
    if if_spherical: if_standard = False
    fastica = FastIca(lpsize*lpsize*nc, only_whiten=ica_only_whiten).to(z.device)
    z = img_to_patches(z, lpsize, lpsize, \
        nc).contiguous()
    z_shape = z.shape
    z = z.view(z.shape[0], -1)
    if if_ica:
        print('fastica')
        z = fastica(z)
    if if_yj:
        print('yj')
        z = yj_transform(z)
    if if_lambt:
        print('lmbt')
        z = lambert_transform(z)
    if if_standard:
        z = normalize(z)
    elif if_spherical:
        z = spherical(z)
    z = z.view(z_shape)
    z = patches_to_img(z, nh, nw, lpsize, lpsize, nc)
    return z

os.makedirs('./figures_sd', exist_ok=True)
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

"""The [pre-trained model](https://huggingface.co/CompVis/stable-diffusion-v1-3-diffusers/tree/main) includes all the components required to setup a complete diffusion pipeline. They are stored in the following folders:
- `text_encoder`: Stable Diffusion uses CLIP, but other diffusion models may use other encoders such as `BERT`.
- `tokenizer`. It must match the one used by the `text_encoder` model.
- `scheduler`: The scheduling algorithm used to progressively add noise to the image during training.
- `unet`: The model used to generate the latent representation of the input.
- `vae`: Autoencoder module that we'll use to decode latent representations into real images.

We can load the components by referring to the folder they were saved, using the `subfolder` argument to `from_pretrained`.
"""

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

"""Now instead of loading the pre-defined scheduler, we load a K-LMS scheduler instead."""

from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

"""Next we move the models to the GPU."""

vae = vae.to(torch_device)
text_encoder = text_encoder
unet = unet.to(torch_device)

"""We now define the parameters we'll use to generate images.

Note that `guidance_scale` is defined analog to the guidance weight `w` of equation (2) in the [Imagen paper](https://arxiv.org/pdf/2205.11487.pdf). `guidance_scale == 1` corresponds to doing no classifier-free guidance. Here we set it to 7.5 as also done previously.

In contrast to the previous examples, we set `num_inference_steps` to 100 to get an even more defined image.
"""

prompt1 = ["a photograph of an astronaut riding a horse"]
# prompt2 = ['A high tech solarpunk utopia in the Amazon rainforest']
prompt2 = ['Skyline of New York City, van Gogh style']
prompt3 = ['concept art painting of a cozy village in a mountainous forested valley, historic english and japanese architecture, realistic, detailed, cel shaded, in the style of makoto shinkai and greg rutkowski and james gurney']

prompts = prompt1 + prompt2 + prompt3

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

batch_size = 1

num_inference_steps = 50            # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

seed= 32
generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise

def sd_generate(latents, prompt):

    """First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model."""

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0].to(torch_device)

    """We'll also get the unconditional text embeddings for classifier-free guidance, which are just the embeddings for the padding token (empty text). They need to have the same shape as the conditional `text_embeddings` (`batch_size` and `seq_length`)"""

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids)[0].to(torch_device)

    """For classifier-free guidance, we need to do two forward passes. One with the conditioned input (`text_embeddings`), and another with the unconditional embeddings (`uncond_embeddings`). In practice, we can concatenate both into a single batch to avoid doing two forward passes."""

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])



    """Cool $64 \times 64$ is expected. The model will transform this latent representation (pure noise) into a `512 × 512` image later on.

    Next, we initialize the scheduler with our chosen `num_inference_steps`.
    This will compute the `sigmas` and exact time step values to be used during the denoising process.
    """

    scheduler.set_timesteps(num_inference_steps)

    """The K-LMS scheduler needs to multiply the `latents` by its `sigma` values. Let's do this here"""

    latents = latents * scheduler.init_noise_sigma

    """We are ready to write the denoising loop."""

    from tqdm.auto import tqdm
    from torch import autocast

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    """We now use the `vae` to decode the generated `latents` back into the image."""

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents).sample

    """And finally, let's convert the image to PIL so we can display or save it."""

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]

from scipy.stats import norm
gaudata = np.random.normal(0, 1.0, 1000)
mu, std = norm.fit(gaudata)
xmin, xmax = -5, 5
def exp_map(latents, file_name, prompt, **args):
    print(f'latents norm = {torch.norm(latents)}')
    img = sd_generate(latents, prompt)
    img.save('./figures_sd/' + file_name + '_before.jpg')
    plt.figure()
    plt.hist(latents.detach().cpu().numpy().ravel(), bins=100, density=True)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.savefig('./figures_sd/hist_' + file_name + '_before.jpg', bbox_inches='tight', dpi=300)
    plt.figure()
    plt.imshow(latents.detach().cpu().numpy()[0,0,...], cmap='gray')
    plt.axis('off')
    plt.savefig('./figures_sd/z_' + file_name + '_before.jpg', bbox_inches='tight', dpi=300)

    with torch.no_grad():
        latents = glayers(latents, **args)
    img = sd_generate(latents, prompt)
    img.save('./figures_sd/' + file_name + '_after.jpg')
    plt.figure()
    plt.hist(latents.detach().cpu().numpy().ravel(), bins=100, density=True)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.savefig('./figures_sd/hist_' + file_name + '_after.jpg', bbox_inches='tight', dpi=300)
    plt.figure()
    plt.imshow(latents.detach().cpu().numpy()[0,0,...], cmap='gray')
    plt.axis('off')
    plt.savefig('./figures_sd/z_' + file_name + '_after.jpg', bbox_inches='tight', dpi=300)
    plt.close('all')

"""Generate the intial random noise."""
latent_h, latent_w = height // 8, width // 8

# latents = torch.randn(
#     (batch_size, unet.in_channels, height // 8, width // 8),
#     generator=generator,
# )
# latents = latents.to(torch_device)
# sd_generate(latents, ['A bucket bag made of blue suede. The bag is decorated with intricate golden paisley patterns. The handle of the bag is made of rubies and pearls.']).save(f'./test.jpg')

# seeds = [123]
seeds = [100, 200, 300, 123]
for pind in range(len(prompts)):
    # if pind == 0 or pind == 2:
    #     continue
    for seed in seeds:
        
        generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
        latents = torch.randn(
          (batch_size, unet.in_channels, height // 8, width // 8),
          generator=generator,
        )
        latents = latents.to(torch_device)
        latents = spherical(latents)
        sd_generate(latents, prompts[pind]).save(f'./figures_sd/seed{seed}_prompt{pind}_1.jpg')
        sd_generate(latents * 0.7, prompts[pind]).save(f'./figures_sd/seed{seed}_prompt{pind}_0.jpg')
        sd_generate(latents * 1.3, prompts[pind]).save(f'./figures_sd/seed{seed}_prompt{pind}_2.jpg')


        # add sin-cos
        generator = torch.manual_seed(seed)   # Seed generator to create the inital latent noise
        latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
        )
        latents = latents.to(torch_device)
        x_sin = torch.sin(2*np.pi/latents.shape[-1] * torch.arange(latents.shape[-1]))
        y_cos = torch.cos(2*np.pi/latents.shape[-2] * torch.arange(latents.shape[-2]))
        xyouter = torch.outer(x_sin, y_cos)
        z = xyouter.reshape(1, 1, latents.shape[-2], latents.shape[-1]).to(torch_device)
        latents = latents + z * 0.5
        latents = spherical(latents)
        exp_map(latents, f'sin_cos_seed{seed}_prompt{pind}', prompts[pind], if_ica=True, ica_only_whiten=False, if_yj=True, if_lambt=True)


        # skewed
        np.random.seed(seed=seed)
        latents = torch.from_numpy(stats.loggamma.rvs(1, size = batch_size*unet.in_channels*latent_h*latent_w)).reshape(batch_size, unet.in_channels, latent_h, latent_w).type(torch.float32).to(torch_device)
        latents = spherical(latents)
        # exp_map(latents, f'skewed_seed{seed}_prompt{pind}', prompts[pind], if_ica=False, if_yj=True, if_lambt=False)
        # we can also use all G-layers at once
        exp_map(latents, f'skewed_seed{seed}_prompt{pind}', prompts[pind], if_ica=False, if_yj=True, if_lambt=True)

        # heavy-tailed
        latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
        )
        latents = latents.to(torch_device)
        latents = latents * torch.exp(0.5 * 0.5 * latents**2)
        latents = spherical(latents)
        # exp_map(latents, f'ht_seed{seed}_prompt{pind}', prompts[pind], if_ica=False, if_yj=False, if_lambt=True)
        # we can also use all G-layers at once
        exp_map(latents, f'ht_seed{seed}_prompt{pind}', prompts[pind], if_ica=True, if_yj=True, if_lambt=True)