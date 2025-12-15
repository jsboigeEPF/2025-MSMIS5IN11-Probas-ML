#!/usr/bin/env python3
"""
DCGAN demo script (PyTorch)

Produces progress images in `progress_images/` using a fixed noise vector
so you can visually track generator improvement across epochs.

Usage (example):
    python dcgan_medical_demo.py --dataset MNIST --epochs 10 --batch_size 128

This script uses MNIST/FashionMNIST as a fast proxy for medical images.
Comments highlight how to adapt to medical image datasets (X-Ray/MRI) later.
"""
import argparse
import os
import random
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm


class Generator(nn.Module):
    """DCGAN Generator

    Input: noise vector (nz)
    Output: image tensor (nc x image_size x image_size) with tanh activation
    """
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        # We'll upscale from 1x1 -> 4x4 -> ... -> image_size
        # Standard DCGAN uses ConvTranspose2d blocks with BatchNorm and ReLU.
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # output size: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """DCGAN Discriminator

    Input: image tensor (nc x image_size x image_size)
    Output: probability (real/fake)
    """
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        # Convolutional blocks with LeakyReLU and BatchNorm (except first)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_data_loader(dataset_name='MNIST', batch_size=128, image_size=64, data_root='./data'):
    # Normalize images to [-1, 1] because the generator uses Tanh output
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name.lower() == 'fashionmnist':
        dataset = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return loader


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')

    # Create output dir
    out_dir = Path('progress_images')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data loader
    dataloader = get_data_loader(args.dataset, args.batch_size, args.image_size, data_root=args.data_root)

    # Model hyperparameters
    nz = args.nz
    nc = args.nc
    ngf = args.ngf
    ndf = args.ndf

    # Create models
    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    netD = Discriminator(nc=nc, ndf=ndf).to(device)

    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(args.num_visualize, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    # Optionally resume from checkpoint (not implemented here, placeholder)

    print(f"Starting Training on device: {device}")

    iters = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit='batch')
        for i, (data, _) in enumerate(progress):
            netD.zero_grad()
            # Train with real images
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake images
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G
            netG.zero_grad()
            label.fill_(real_label)  # want generator to produce images classified as real
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            iters += 1
            if i % args.log_interval == 0:
                progress.set_postfix({'errD': errD.item(), 'errG': errG.item(), 'D(x)': D_x, 'D(G(z))': D_G_z2})

            # Optionally save intermediate grids every X iterations
            if args.save_every_iters > 0 and iters % args.save_every_iters == 0:
                with torch.no_grad():
                    fake_fixed = netG(fixed_noise).detach().cpu()
                save_image(fake_fixed, out_dir / f'iter_{iters:06d}.png', normalize=True, nrow=8)

        # End of epoch: save grid using fixed noise
        with torch.no_grad():
            fake_fixed = netG(fixed_noise).detach().cpu()
        epoch_file = out_dir / f'epoch_{epoch:03d}.png'
        save_image(fake_fixed, epoch_file, normalize=True, nrow=8)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} finished in {epoch_time:.1f}s — saved {epoch_file}")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s. Final models saved in memory.")

    # Save the final generator and discriminator for later fine-tuning / adaptation
    torch.save(netG.state_dict(), 'netG_final.pth')
    torch.save(netD.state_dict(), 'netD_final.pth')


def parse_args():
    parser = argparse.ArgumentParser(description='DCGAN demo (MNIST proxy for medical images)')
    parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST or FashionMNIST')
    parser.add_argument('--data_root', type=str, default='./data', help='path to dataset root')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--image_size', type=int, default=64, help='spatial size of training images')
    parser.add_argument('--nc', type=int, default=1, help='number of channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='generator feature maps')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator feature maps')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--num_visualize', type=int, default=64, help='number of images to generate for visualization')
    parser.add_argument('--save_every_iters', type=int, default=0, help='save intermediate grids every N iterations (0 to disable)')
    parser.add_argument('--log_interval', type=int, default=100, help='progress log interval (batches)')
    parser.add_argument('--force_cpu', action='store_true', help='force CPU even if CUDA available')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Run training
    train(args)

    # Explanatory comments about usage for medical augmentation:
#
# In medical imaging, a trained GAN (like this DCGAN) can be used to produce synthetic
# images to augment datasets. For example, by conditioning (cGAN) or carefully selecting
# latent vectors, researchers can generate more examples of rare pathologies to help
# classifiers see more variety during training. Important caveats:
# - Generated images must be validated by clinicians before use.
# - GANs can produce artifacts; post-processing and quality filtering are necessary.
# - For higher-resolution medical images, use deeper architectures, multi-scale training,
#   or progressive growing (e.g., Progressive GANs) and ensure appropriate normalization
#   and domain-specific augmentations.
