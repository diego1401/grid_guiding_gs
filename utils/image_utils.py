#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def fourier_ratio_on_image(img1, img2):
    '''
    This gives us the ratio between the real part (signal strength) of the fourier coefficients.
    This way a low-freq / high-freq image < 1. A value smaller than 1 should show that img1 is then
    lower frequency than img2.
    '''
    return torch.abs(torch.real(torch.fft.fftn(img1)))/(torch.abs(torch.real(torch.fft.fftn(img2))) + 1e-6)
