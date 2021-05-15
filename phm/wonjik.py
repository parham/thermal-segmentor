
""" 
    @name           Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering   
    @journal        IEEE Transactions on Image Processing
    @year           2020
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
    @citation       Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    @info           The code is adopted from the mentioned repo.

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
import cv2
import sys
import numpy as np
import torch.nn.init
import random

from phm import load_config, Segmentator


class Wonjik2020Net (nn.Module):
    """
        Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    """

    def __init__(self, net_config, num_dim):
        super(Wonjik2020Net, self).__init__()

        # Set the model's config based on provided configuration
        self.config = net_config
        nChannel = self.config['num_channels']
        nConv = self.config['num_conv_layers']

        # First convolutional layer
        self.conv1 = nn.Conv2d(num_dim, nChannel,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        # Feature space including multiple convolutional layers
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        # The number of convolutional layers are determined based on "nCov" parameters.
        for i in range(nConv-1):
            tmpConv = nn.Conv2d(nChannel, nChannel,
                                kernel_size=3, stride=1, padding=1)
            tmpBatch = nn.BatchNorm2d(nChannel)
            self.conv2.append(tmpConv)
            self.bn2.append(tmpBatch)
        # The reference normalization for extracting class labels
        self.conv3 = nn.Conv2d(
            nChannel, nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        nConv = self.config['num_conv_layers']

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(nConv-1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)

        x = self.conv3(x)
        x = self.bn3(x)
        return x

class Wonjik2020Segmentator(Segmentator):
    def __init__(self, seg_config) -> None:
        super().__init__(seg_config['segmentation'])
        self.model_config = seg_config['model']
    
    def segment(self,img):
        nChannel = self.model_config['num_channels']

        img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])
        data = torch.from_numpy(img_data)
        if self.use_cuda:
            data = data.cuda()
        data = Variable(data)
        # Create an instance of the model and set it to learn
        model = Wonjik2020Net(self.model_config, data.size(1))
        if self.use_cuda:
            model.cuda()
        model.train()
        # similarity loss definition
        loss_fn = nn.CrossEntropyLoss()
        # continuity loss definition
        loss_hpy = torch.nn.L1Loss(size_average = True)
        loss_hpz = torch.nn.L1Loss(size_average = True)

        HPy_target = torch.zeros(img.shape[0]-1, img.shape[1], nChannel)
        HPz_target = torch.zeros(img.shape[0], img.shape[1]-1, nChannel)
        if self.use_cuda:
            HPy_target = HPy_target.cuda()
            HPz_target = HPz_target.cuda()
        
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        label_colours = np.random.randint(255,size=(100,3))

        seg_result = None
        seg_num_classes = 0
        for batch_idx in range(self.iteration):
            # forwarding
            optimizer.zero_grad()
            output = model(data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)

            outputHP = output.reshape( (img.shape[0], img.shape[1], nChannel))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy,HPy_target)
            lhpz = loss_hpz(HPz,HPz_target)

            _, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))

            seg_result = im_target
            seg_num_classes = nLabels

            if self.visualize:
                im_target_rgb = np.array([label_colours[ c % nChannel ] for c in im_target])
                im_target_rgb = im_target_rgb.reshape( img.shape ).astype( np.uint8 )
                cv2.imshow( "output", im_target_rgb )
                cv2.waitKey(10)

            # loss calculation
            loss = self.similarity_loss_ssize * loss_fn(output, target) + self.continuity_loss_ssize * (lhpy + lhpz)
            loss.backward()
            optimizer.step()

            print (batch_idx, '/', self.iteration, '|', ' label num :', nLabels, ' | loss :', loss.item())

            if nLabels <= self.min_classes:
                print ("nLabels", nLabels, "reached min_classes", self.min_classes, ".")
                break
    
        return seg_result, seg_num_classes