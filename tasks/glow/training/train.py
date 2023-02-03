# modified from https://raw.githubusercontent.com/rosinality/glow-pytorch/master/train.py

from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import os, sys

import argparse
sys.path.append('../../../')

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dgminv.networks.models_cond_glow import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=64, type=int, help="batch size")
parser.add_argument("--epochs", default=1000, type=int, help="maximum number of epochs")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=128, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    return loader


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, model, optimizer):
    data_loader = sample_data(args.path, args.batch, args.img_size)
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    model.train()
    with tqdm(range(args.epochs)) as pbar:
        for epoch in pbar:
            num_step = 0
            for i, data in enumerate(data_loader):
                num_step += 1
                image = data[0].to(device)

                image = image * 255

                if args.n_bits < 8:
                    image = torch.floor(image / 2 ** (8 - args.n_bits))

                image = image / n_bins - 0.5

                image = image + torch.rand_like(image) / n_bins

                if epoch == 0 and num_step == 1:
                    with torch.no_grad():
                        log_p, logdet, _ = model(image)
                        continue
                else:
                    log_p, logdet, _ = model(image)

                loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
                model.zero_grad()
                loss.backward()

                # grad procesing
                torch.nn.utils.clip_grad_value_(model.parameters() , 5)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters() , 100)

                # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
                warmup_lr = args.lr
                optimizer.param_groups[0]["lr"] = warmup_lr
                optimizer.step()

                if i % 10 == 0:
                    pbar.set_description(
                        f"Epoch: {epoch}; i: {i+1}; Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}"
                    )

                if i % 100 == 0:
                    with torch.no_grad():
                        generated_imgs,_,_ = model_single.reverse(z_sample)
                        utils.save_image(
                            generated_imgs.cpu().data,
                            f"samples/{str(num_step + 1).zfill(6)}.png",
                            normalize=True,
                            nrow=10,
                            range=(-0.5, 0.5),
                        )

            if epoch % 1 == 0:
                torch.save(
                    model.module.state_dict(), f"checkpoints/Flow_net_epoch{epoch}.pt"
                )


if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    args = parser.parse_args()

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu, cond=False
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    train(args, model, optimizer)