
""" 
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University
"""

import logging
from dotmap import DotMap

import torch
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import torch.nn.functional as F

import time
import cv2
import numpy as np
from skimage import segmentation
from phm import running_time
from tkinter import Variable
from comet_ml import Experiment

from phm.core import Segmentor

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
        self.cls = nn.Softmax2d()

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
        self.conv = nn.Conv2d(
            in_channel, out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
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
                 output_padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        return x

class phmAutoencoderModule (nn.Module):
    def __init__(self, config : DotMap, num_dim):
        super(phmAutoencoderModule, self).__init__()
        # Set the model's config based on provided configuration
        self.config = config
        self.nChannel = self.config.num_channels
        # Part 01 : the feature extraction
        self.part01 = nn.ModuleList([
            Feature(num_dim, self.nChannel,
                kernel_size=self.config.part01_kernel_size,
                stride=self.config.part01_stride,
                padding=self.config.part01_padding),
            Feature(self.nChannel, self.nChannel,
                kernel_size=self.config.part01_kernel_size,
                stride=self.config.part01_stride,
                padding=self.config.part01_padding),
            Feature(self.nChannel, self.nChannel,
                kernel_size=self.config.part01_kernel_size,
                stride=self.config.part01_stride,
                padding=self.config.part01_padding),
        ])
        # Feature space including multiple convolutional layers
        # Part 02 : Auto-Encoder
        self.encoder = []
        inc_ch = self.nChannel
        # Encoder
        for i in range(self.config.part02_num_layer):
            tmp = Feature(inc_ch, inc_ch * 2,
                kernel_size=self.config.part02_kernel_size,
                stride=self.config.part02_stride,
                padding=self.config.part02_padding)
            inc_ch *= 2
            self.encoder.append(tmp)
        self.encoder = nn.ModuleList(self.encoder)
        # Decoder
        self.decoder = []
        for i in range(self.config.part02_num_layer):
            tmp = TransposeFeature(int(inc_ch), int(inc_ch / 2),
                                   kernel_size=self.config.part02_kernel_size,
                                   stride=self.config.part02_stride,
                                   padding=self.config.part02_padding,
                                   output_padding=self.config.part02_output_padding)
            inc_ch /= 2
            self.decoder.append(tmp)
        self.decoder = nn.ModuleList(self.decoder)
        # Part 03 : the reference normalization for extracting class labels
        self.part03 = nn.ModuleList([
            Feature(self.nChannel, self.nChannel,
                    kernel_size=self.config.part03_kernel_size,
                    stride=self.config.part03_stride,
                    padding=self.config.part03_padding),
            Feature(self.nChannel, self.nChannel,
                    kernel_size=self.config.part03_kernel_size,
                    stride=self.config.part03_stride,
                    padding=self.config.part03_padding),
        ])
        self.part04 = nn.ModuleList([
            Feature(self.nChannel, self.nChannel,
                kernel_size=5,
                stride=1,
                padding=2),
            Feature(self.nChannel, self.nChannel,
                kernel_size=5,
                stride=1,
                padding=2),
            Feature(self.nChannel, self.nChannel,
                kernel_size=3,
                stride=1,
                padding=1),
            Feature(self.nChannel, self.nChannel,
                kernel_size=5,
                stride=1,
                padding=2),
            Feature(self.nChannel, self.nChannel,
                kernel_size=5,
                stride=1,
                padding=2),
        ])
        # Part 04 : the final classification
        self.classify = Classifier(self.nChannel, self.nChannel,
            kernel_size=self.config.part04_kernel_size,
            stride=self.config.part04_stride,
            padding=self.config.part04_padding)

    def forward(self, x):
        # Part 01
        for sp in self.part01:
            x = sp(x)
        # Part 02
        # Encoder
        en_out = list()
        for sp in self.encoder:
            x = sp(x)
            en_out.append(x)
        # Decoder
        first_layer = True
        for sp in self.decoder:
            tmp = en_out.pop()
            if first_layer:
                first_layer = False
            else:
                # import pdb; pdb.set_trace()
                x = torch.cat((x,tmp), dim=-1)
            x = sp(x)

        avg0 = nn.AvgPool2d(4, stride=1, padding=1)
        x = avg0(x)
        # Part 03
        for sp in self.part03:
            x = sp(x)
        #############
        avg1 = nn.AvgPool2d(3, stride=1, padding=1)
        # max1 = nn.MaxPool2d(4, stride=1)
        # up1 = nn.Upsample(x.shape[2:])
        x = avg1(x)
        # x = max1(x)
        # x = up1(x)
        # for sp in self.part04:
        #     x = sp(x)
        # Part 04
        x = self.classify(x)

        return x

class phmAutoencoderSegmentor(Segmentor):
    
    def __init__(self, 
        config: DotMap, 
        experiment: Experiment = None, 
        optimizer=None, 
        use_cuda: bool = True) -> None:

        super().__init__(config, 
            experiment, None, optimizer, 
            torch.nn.CrossEntropyLoss(), use_cuda)
        # Number of channels
        self.nChannel = self.config.model.num_channels
        # Label Colors
        self.label_colours = np.random.randint(255,size=(100,3))
        # similarity loss definition
        self.loss_fn = nn.CrossEntropyLoss()
        # continuity loss definition
        self.loss_hpy = torch.nn.SmoothL1Loss(size_average=True)
        self.loss_hpz = torch.nn.SmoothL1Loss(size_average=True)

    def calc_loss(self, img, output, target):
        img_w = img.shape[0]
        img_h = img.shape[1]
        outputHP = output.reshape((img_w, img_h, self.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = self.loss_hpy(HPy, self.HPy_target)
        lhpz = self.loss_hpz(HPz, self.HPz_target)
        # loss calculation
        # return self.config.segmentation.similarity_loss_ssize * self.loss_fn(output, target) + self.config.segmentation.continuity_loss_ssize * (lhpy + lhpz)
        return self.config.segmentation.similarity_loss_ssize * self.loss_fn(output.view(-1, self.nChannel), target.view(-1)) + self.config.segmentation.continuity_loss_ssize * (lhpy + lhpz)

    def _segment(self, img):
        img_w = img.shape[0]
        img_h = img.shape[1]
        img_dim = img.shape[2]
        # Convert image to numpy data
        img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])
        data = torch.from_numpy(img_data)
        if self.use_cuda:
            data = data.cuda()

        self.HPy_target = torch.zeros(
            img_w-1, img_h, self.nChannel)
        self.HPz_target = torch.zeros(
            img_w, img_h-1, self.nChannel)
        if self.use_cuda:
            self.HPy_target = self.HPy_target.cuda()
            self.HPz_target = self.HPz_target.cuda()

        if self.model is None:
            self.model = phmAutoencoderModule(self.config.model, img_dim)

        if self.optimizer is None:       
            lr = self.config.segmentation.learning_rate
            momentum = self.config.segmentation.momentum
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        #####################################

        # Create an instance of the model and set it to learn
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.train()

        seg_result = None
        seg_num_classes = 0
        seg_step_time = 0

        with self.experiment.train():
            for step in range(self.config.segmentation.iteration):
                t = time.time()
                self.optimizer.zero_grad()
                output = self.model(data)[0,:,0:img_w,0:img_h]

                output_orig = output.permute(1, 2, 0).contiguous()
                output = output_orig.view(-1, self.nChannel)

                _, target = torch.max(output, 1)
                im_target = target.data.cpu().numpy()
                nLabels = len(np.unique(im_target))

                seg_result = im_target.reshape(img.shape[0:2])
                seg_num_classes = nLabels

                target = torch.from_numpy(im_target)
                if self.use_cuda:
                    target = target.cuda()

                loss = self.calc_loss(img, output, target)
                loss.backward()
                self.optimizer.step()

                logging.info(f'{step} / {self.config.segmentation.iteration} : {nLabels} , {loss.item()}')

                step_time = time.time() - t
                seg_step_time += step_time
                
                self.experiment.log_metrics({
                    'step_time' : step_time,
                    'class_count' : nLabels,
                    'loss' : loss
                }, step=step, epoch=1)
                self.experiment.log_image(seg_result,name='steps',step=step)
        
                if nLabels <= self.config.segmentation.min_classes:
                    logging.info(f'Number of labels has reached {nLabels}.')
                    break

        return seg_result, {
            'iteration_time' : seg_step_time,
            'num_classes' : seg_num_classes
        }