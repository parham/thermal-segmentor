
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @description Adopted from https://github.com/Po-Hsun-Su/pytorch-ssim
"""

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from lemanchot.loss.core import BaseLoss, loss_register

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    
    mu1 = F.conv2d(img1, window, padding = window_size // 2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size // 2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size // 2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size // 2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size // 2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

@loss_register('ssim_loss')
class SSIMLoss(BaseLoss):
    def __init__(self, name : str, config) -> None:
        """
        Args:
            name (str): _description_
            config (Dict): _description_
        Parameters:
            window_size: window size
            size_average: the size of averaging window
            num_channels: the number of channels
        """
        super().__init__(
            name=name,
            config=config
        )
        self.channel = self.num_channels
        self.window = create_window(self.window_size, self.channel)

    def forward(self, img1, img2):
        channel = img1.shape[1]

        if channel == self.channel and \
           self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
