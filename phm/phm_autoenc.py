
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
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random

from phm import load_config, Segmentator, Wonjik2020Segmentator


class Classifier(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 stride=1,
                 padding=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Feature(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 stride=1,
                 padding=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding
                              )
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        return x


class TransposeFeature(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 output_padding=0
                 ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=output_padding
                                       )
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        return x


class UnsupervisedNN (nn.Module):
    def __init__(self, net_config, num_dim):
        super(UnsupervisedNN, self).__init__()
        # Set the model's config based on provided configuration
        self.config = net_config
        self.nChannel = self.config['num_channels']
        # Part 01 : the feature extraction
        self.part01 = nn.ModuleList([
            Feature(num_dim, self.nChannel,
                    kernel_size=self.config['part01_kernel_size'],
                    stride=self.config['part01_stride'],
                    padding=self.config['part01_padding']
                    ),
            Feature(self.nChannel, self.nChannel,
                    kernel_size=self.config['part01_kernel_size'],
                    stride=self.config['part01_stride'],
                    padding=self.config['part01_padding']
                    ),
            Feature(self.nChannel, self.nChannel,
                    kernel_size=self.config['part01_kernel_size'],
                    stride=self.config['part01_stride'],
                    padding=self.config['part01_padding']
                    ),
        ])
        # Feature space including multiple convolutional layers
        # Part 02 : Auto-Encoder
        self.encoder = []
        inc_ch = self.nChannel
        # Encoder
        for i in range(self.config['part02_num_layer']):
            tmp = Feature(inc_ch, inc_ch * 2,
                          kernel_size=self.config['part02_kernel_size'],
                          stride=self.config['part02_stride'],
                          padding=self.config['part02_padding']
                          )
            inc_ch *= 2
            self.encoder.append(tmp)
        self.encoder = nn.ModuleList(self.encoder)
        # Decoder
        self.decoder = []
        for i in range(self.config['part02_num_layer']):
            tmp = TransposeFeature(int(inc_ch), int(inc_ch / 2),
                                   kernel_size=self.config['part02_kernel_size'],
                                   stride=self.config['part02_stride'],
                                   padding=self.config['part02_padding'],
                                   output_padding=self.config['part02_output_padding'])
            inc_ch /= 2
            self.decoder.append(tmp)
        self.decoder = nn.ModuleList(self.decoder)
        # Part 03 : the reference normalization for extracting class labels
        self.part03 = nn.ModuleList([
            Feature(self.nChannel, self.nChannel,
                    kernel_size=self.config['part03_kernel_size'],
                    stride=self.config['part03_stride'],
                    padding=self.config['part03_padding']
                    ),
            Feature(self.nChannel, self.nChannel,
                    kernel_size=self.config['part03_kernel_size'],
                    stride=self.config['part03_stride'],
                    padding=self.config['part03_padding']
                    ),
        ])
        # Part 04 : the final classification
        self.classify = Classifier(self.nChannel, self.nChannel,
                                 kernel_size=self.config['part04_kernel_size'],
                                 stride=self.config['part04_stride'],
                                 padding=self.config['part04_padding']
                                 )

    def forward(self, x):
        # Part 01
        for sp in self.part01:
            x = sp(x)
        # Part 02
        # Encoder
        for sp in self.encoder:
            x = sp(x)
        # Decoder
        for sp in self.decoder:
            x = sp(x)
        # Part 03
        for sp in self.part03:
            x = sp(x)
        #############
        # avg1 = nn.AvgPool2d(4, stride=1)
        # # max1 = nn.MaxPool2d(4, stride=1)
        # up1 = nn.Upsample(x.shape[2:])
        # x = avg1(x)
        # # x = max1(x)
        # x = up1(x)
        # Part 04
        x = self.classify(x)

        return x


class UnsupervisedSegmentor(Wonjik2020Segmentator):
    def __init__(self,
                 seg_config,
                 model=None,
                 optimizer=None,
                 ) -> None:
        super().__init__(seg_config=seg_config, model=model, optimizer=optimizer)
        # similarity loss definition
        self.loss_fn = nn.CrossEntropyLoss()
        # continuity loss definition
        self.loss_hpy = torch.nn.SmoothL1Loss(size_average=True)
        self.loss_hpz = torch.nn.SmoothL1Loss(size_average=True)

