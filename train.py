from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import define_D, define_G, GANLoss, get_schedulers
from dataset import ImageDataset
from options import Options

def main(opt):
    # if opt.cuda and not torch.cuda.is_available():
    #     raise Exception('No GPU found, please run without --cuda')
    
    cudnn.benchmark = True
    # torch.manual_seed(opt.seed)

    print('--------------------Loading datasets--------------------')

    # Training data transforms
    train_transforms = transforms.Compose([
        transforms.Resize((opt.load_size, opt.load_size), transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Define train dataset and dataloader
    train_dataset = ImageDataset(opt.dataroot, transforms=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=True)
    print(f'Create dataset successfully and have {len(train_dataset)} images in train dataset')
    print('----------------------------------------------------------')

    print('--------------------Network initialized-------------------')

    # Build generator and discriminator network
    netG = define_G(opt.input_nc, opt.output_nc, ngf=opt.ngf, netG=opt.netG, norm=opt.norm, use_dropout=opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
    print(f'Network G Total number of parameters: {sum(p.numel() for p in netG.parameters())}')

    netD = define_D(opt.input_nc + opt.output_nc, ndf=opt.ndf, netD=opt.netD, n_layers_D=opt.n_layers_D, norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
    print(f'Network D Total number of parameters: {sum(p.numel() for p in netD.parameters())}')
    print('-----------------------------------------------------------')

    # Define loss
    criterionGAN = GANLoss(gan_mode=opt.gan_mode)
    criterionL1 = nn.L1Loss()

    # Define optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    netG_scheduler = get_schedulers(optimizerG, opt)
    netD_scheduler = get_schedulers(optimizerD, opt)
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.in_epochs_decay +1):
        for _, data in enumerate(train_loader, 1):
            real_A, real_B = data[0], data[1]
            fake_B = netG(real_A)

            # Train the discriminator
            optimizerD.zero_grad()

            # Fake, stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = netD(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False)

            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, True)

            # Combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()

            # Update weights
            optimizerD.step()

            # Train the generator
            optimizerG.zero_grad()

            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = netD(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B) * opt.lambda_L1

            # Combine loss and calculate gradients
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()

            # Update weights
            optimizerG.step()

if __name__ == '__main__':
    opt = Options().parser()
    main(opt)