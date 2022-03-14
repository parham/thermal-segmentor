
""" 
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University

    @name           Unsupervised Image Segmentation by Backpropagation
    @journal        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    @year           2018
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation
    @citation       Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.
    @info           the code is based on the implementation presented in the mentioned repository.
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

class Kanezaki2018Module (nn.Module):

    def __init__(self, config : DotMap, num_dim):
        super(Kanezaki2018Module, self).__init__()
        # Set the model's config based on provided configuration
        self.config = config
        nChannel = self.config.num_channels
        nConv = self.config.num_conv_layers

        self.conv1 = nn.Conv2d(
            num_dim, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append(nn.Conv2d(
                nChannel, nChannel,
                    kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannel))
        self.conv3 = nn.Conv2d(nChannel, nChannel, 
            kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        nConv = self.config.num_conv_layers

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class Kanezaki2018Segmentator(Segmentor):

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
        self.l_inds = None

    def calc_loss(self, output, target):
        # superpixel refinement
        im_target = target.data.cpu().numpy()
        for i in range(len(self.l_inds)):
            labels_per_sp = im_target[self.l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[self.l_inds[i]] = u_labels_per_sp[np.argmax(hist)]

        return self.loss_fn(output, target)

    def _segment(self, img) -> dict:
        img_w = img.shape[0]
        img_h = img.shape[1]
        img_dim = img.shape[2]
        # Convert image to numpy data
        img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])
        data = torch.from_numpy(img_data)
        if self.use_cuda:
            data = data.cuda()

        ####### Initialization steps #######
        # SLIC : segment the image using SLIC algorithm
        labels = segmentation.slic(img, compactness=self.config.segmentation.compactness, 
            n_segments=self.config.segmentation.superpixel_regions)
        # flatten the image
        labels = labels.reshape(img_w * img_h)
        # Extract the unique label
        u_labels = np.unique(labels)
        self.l_inds = []
        for i in range(len(u_labels)):
            self.l_inds.append(np.where(labels == u_labels[i])[0])

        # Create an instance of the model
        if self.model is None:
            self.model = Kanezaki2018Module(self.config.model, img_dim) 

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

                loss = self.calc_loss(output, target)
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
