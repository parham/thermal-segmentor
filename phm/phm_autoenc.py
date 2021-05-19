
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
        self.part04 = Classifier(self.nChannel, self.nChannel,
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
        # x = max1(x)
        # x = up1(x)
        # Part 04
        x = self.part04(x)

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


class PHMAutoencoder01 (nn.Module):

    def __init__(self, net_config, num_dim):
        super(PHMAutoencoder01, self).__init__()

        # Set the model's config based on provided configuration
        self.config = net_config
        nChannel = self.config['num_channels']

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            num_dim, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        # Second Layer
        self.conv2 = nn.Conv2d(
            nChannel, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(nChannel)
        # Third Layer
        self.conv3 = nn.Conv2d(
            nChannel, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(nChannel)

        self.conv31 = nn.Conv2d(
            nChannel, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(nChannel)
        # Feature space including multiple convolutional layers
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(nChannel, nChannel * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(nChannel * 2),
            nn.Conv2d(nChannel * 2, nChannel * 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(nChannel * 4),
            nn.Conv2d(nChannel * 4, nChannel * 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(nChannel * 8),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nChannel * 8, nChannel * 4, 3),
            nn.ReLU(),
            nn.BatchNorm2d(nChannel * 4),
            nn.ConvTranspose2d(nChannel * 4, nChannel * 2, 3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(nChannel * 2),
            nn.ConvTranspose2d(nChannel * 2, nChannel, 3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(nChannel),
            nn.AvgPool2d(4, stride=1)
        )
        # The reference normalization for extracting class labels
        self.conv4 = nn.Conv2d(
            nChannel, nChannel, kernel_size=3, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(nChannel)

        self.conv5 = nn.Conv2d(
            nChannel, nChannel, kernel_size=3, stride=1, padding=2)
        self.bn5 = nn.BatchNorm2d(nChannel)
        # self.conv5 = nn.Conv2d(nChannel, 3, kernel_size=5, stride=1, padding=2)
        # self.bn5 = nn.BatchNorm2d(3)

        # self.conv6 = nn.Conv2d(
        #     3, 3, kernel_size=1, stride=1, padding=0)
        # self.bn6 = nn.BatchNorm2d(3)

        self.conv6 = nn.Conv2d(
            nChannel, nChannel, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.conv31(x)
        x = F.relu(x)
        x = self.bn31(x)

        x = self.encoder(x)
        x = self.decoder(x)
        max1 = nn.MaxPool2d(4, stride=4)
        up1 = nn.Upsample(x.shape[2:])
        x = max1(x)
        x = up1(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.bn6(x)

        return x


class PHMAutoencoder01Segmentator(Segmentator):
    def __init__(self, seg_config) -> None:
        super().__init__(seg_config['segmentation'])
        self.model_config = seg_config['model']

    def segment(self, img):
        nChannel = self.model_config['num_channels']

        img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])
        data = torch.from_numpy(img_data)
        if self.use_cuda:
            data = data.cuda()
        data = Variable(data)
        # Create an instance of the model and set it to learn
        model = PHMAutoencoder01(self.model_config, data.size(1))
        if self.use_cuda:
            model.cuda()
        model.train()
        # similarity loss definition
        loss_fn = nn.CrossEntropyLoss()
        # continuity loss definition
        # loss_hpy = torch.nn.L1Loss(size_average = True)
        # loss_hpz = torch.nn.L1Loss(size_average = True)
        loss_hpy = torch.nn.SmoothL1Loss(size_average=True)
        loss_hpz = torch.nn.SmoothL1Loss(size_average=True)

        HPy_target = torch.zeros(img.shape[0]-1, img.shape[1], nChannel)
        HPz_target = torch.zeros(img.shape[0], img.shape[1]-1, nChannel)
        if self.use_cuda:
            HPy_target = HPy_target.cuda()
            HPz_target = HPz_target.cuda()

        optimizer = optim.SGD(model.parameters(),
                              lr=self.learning_rate, momentum=0.9)
        label_colours = np.random.randint(255, size=(nChannel, 3))

        img_w = img.shape[0]
        img_h = img.shape[1]
        seg_result = None
        seg_num_classes = 0

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        for batch_idx in range(self.iteration):
            # forwarding
            optimizer.zero_grad()
            output = model(data)[0, :, 0:img_w, 0:img_h]

            output_orig = output.permute(1, 2, 0).contiguous()
            output = output_orig.view(-1, nChannel)

            # outputHP = output.reshape((img.shape[0], img.shape[1], nChannel))
            outputHP = output.reshape((img.shape[0], img.shape[1], nChannel))

            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)

            _, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))

            seg_result = im_target
            seg_num_classes = nLabels

            if self.visualize:
                im_target_rgb = np.array(
                    [label_colours[c % nChannel] for c in im_target])
                im_target_rgb = im_target_rgb.reshape(
                    img.shape).astype(np.uint8)
                cv2.imshow("output", im_target_rgb)
                cv2.waitKey(10)

            # loss calculation
            loss = self.similarity_loss_ssize * \
                loss_fn(output, target) + \
                self.continuity_loss_ssize * (lhpy + lhpz)
            loss.backward()
            optimizer.step()

            print(batch_idx, '/', self.iteration, '|',
                  ' label num :', nLabels, ' | loss :', loss.item())

            if nLabels <= self.min_classes:
                print("nLabels", nLabels, "reached min_classes",
                      self.min_classes, ".")
                break

        return seg_result, seg_num_classes
