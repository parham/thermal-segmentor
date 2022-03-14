
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
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.utils import _pair, _quadruple


import cv2
import sys
import numpy as np
import torch.nn.init
import random
import time

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
        # self.cls = nn.Softmax2d()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.cls(x)

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


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
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
                output_padding=self.config['part02_output_padding']
            )
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
        # Part 04
        avg1 = nn.AvgPool2d(5, stride=1)
        x = avg1(x)
        x = self.classify(x)
        #############
        # avg1 = MedianPool2d(4, stride=1)
        # # max1 = nn.MaxPool2d(4, stride=1)
        # up1 = nn.Upsample(x.shape[2:])
        # # x = max1(x)
        # x = up1(x)

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

    def calc_loss(self, img, output, target):
        outputHP = output
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = self.loss_hpy(HPy, self.HPy_target)
        lhpz = self.loss_hpz(HPz, self.HPz_target)
        # loss calculation
        return self.similarity_loss_ssize * self.loss_fn(output.view(-1, self.nChannel), target.view(-1)) + self.continuity_loss_ssize * (lhpy + lhpz)

    def segment_(self, img) -> dict:
        # Convert image to numpy data
        img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])
        data = torch.from_numpy(img_data)
        if self.use_cuda:
            data = data.cuda()
        data = Variable(data)

        self.pre_segment(img)

        # Create an instance of the model and set it to learn
        if self.use_cuda:
            self.model.cuda()
        self.model.train()

        result = {}
        seg_result = None
        seg_num_classes = 0
        seg_step_time = 0
        img_w = img.shape[0]
        img_h = img.shape[1]

        if self.visualize:
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)  

        for batch_idx in range(self.iteration):
            t = time.time()
            self.optimizer.zero_grad()
            output = self.model(data)[0,:,0:img_w,0:img_h]
            output = output.permute(1, 2, 0).contiguous()

            _, target = torch.max(output, 2)

            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target.flatten()))

            seg_result = im_target
            seg_num_classes = nLabels

            if self.visualize:
                im_target_rgb = np.dstack([im_target, im_target, im_target]).astype(np.uint8)
                cv2.imshow("output", im_target_rgb)
                cv2.waitKey(10)

            target = torch.from_numpy(im_target)
            if self.use_cuda:
                target = target.cuda()

            target = Variable(target)
            loss = self.calc_loss(img, output, target)

            loss.backward()
            self.optimizer.step()

            print (batch_idx, '/', self.iteration, ':', nLabels, loss.item())

            seg_step_time += time.time() - t
            if nLabels <= self.min_classes:
                print ("nLabels", nLabels, "reached minLabels", self.min_classes, ".")
                break
        
        result['iteration_time'] = seg_step_time
        result['segmentation'] = seg_result
        result['num_classes'] = seg_num_classes
        return result