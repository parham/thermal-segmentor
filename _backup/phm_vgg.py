
""" 
    @name       The use of Autoencoder in the case of unsupervised segmentation
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torchvision.models import vgg19_bn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random

from phm import load_config, Segmentator, Wonjik2020Segmentator

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, in_c, kernel_size=2, stride=2),
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(),
            nn.Conv2d(out_c//2, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class VGGBasedAutoEncoder (nn.Module):

    def __init__(
        self,
        net_config, 
        num_dim,
        depth: int=5,
        freeze_base_model: bool=False
    ) -> None:
        super().__init__()
        
        self.config = net_config
        self.nChannel = self.config['num_channels']

        self.depth = depth
        self._enc_channels = [64, 128, 256, 512, 512][:self.depth]
        self._dec_channels = self._enc_channels[::-1] + [128]

        # VGG19
        self.encoder = self.load_base_model(freeze_base_model)
        self.encoder = nn.ModuleList(self.encoder)

        self.decoder = [Up(self._dec_channels[0], self._dec_channels[1])]

        in_c = self._dec_channels[1]
        for channel in self._dec_channels[2:]:
            self.decoder += [Up(in_c*2, channel)]
            in_c = channel
        
        self.decoder = nn.ModuleList(self.decoder)

        self.classifier = nn.Sequential(
            nn.Conv2d(128, self.nChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nChannel),
            nn.ReLU(),
            nn.Conv2d(self.nChannel, self.nChannel, kernel_size=1, padding=0)
        )

    def load_base_model(self, freeze: bool=True) -> list:
        
        # Getting only the convolution layer from VGG19_BN
        model = list(vgg19_bn(pretrained=True).features.children())

        blocks = [(0, 7), (7, 14), (14, 27), (27, 40), (40, 53)]
        features = [nn.Sequential(*model[i:j]) for i, j in blocks[:self.depth]]
        
        # Freeze parameters
        if freeze:
            for model in features:
                for param in model.parameters():
                    param.requires_grad = False

        return features

    def forward(self, input):
        
        encoded = list()
        encoded.append(input)

        tmp = input
        for i, layer in enumerate(self.encoder):
            tmp = layer(tmp)
            encoded.insert(0,tmp)
            # encoded.append(l(encoded[i]))

        decoded = list()
        tmp = self.decoder[0](tmp)
        decoded.append(tmp)
        for i, layer in enumerate(self.decoder[1:]):
            pass_data  = encoded[i+1]
            dcat = torch.cat((tmp, pass_data), dim=1)
            tmp = layer(dcat)
            decoded.append(tmp)
        
        res = self.classifier(tmp)
        return res
