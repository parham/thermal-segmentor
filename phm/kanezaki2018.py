
""" 
    @author        Parham Nooralishahi
    @email         parham.nooralishahi@gmail.com
    @professor     Professor Xavier Maldague
    @organization: Laval University
"""

import functools
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Metric

from ignite.engine import Engine, EventEnum

import time
import numpy as np
from skimage import segmentation
from comet_ml import Experiment

from phm.core import load_config
from phm.segment import KanezakiIterativeSegmentor, phmLoss

class Kanezaki2018Events(EventEnum):
    INTERNAL_TRAIN_LOOP_COMPLETED = 'internal_train_loop_completed'

class Kanezaki2018Module(nn.Module):
    """ Implementation of the model presented in:
    @name           Unsupervised Image Segmentation by Backpropagation
    @journal        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    @year           2018
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation
    @citation       Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.
    """

    def __init__(self, num_dim, num_channels: int = 100, num_convs: int = 3):
        super(Kanezaki2018Module, self).__init__()

        self.num_convs = num_convs
        self.num_channel = num_channels

        self.conv1 = nn.Conv2d(
            num_dim, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(num_convs-1):
            self.conv2.append(nn.Conv2d(
                num_channels, num_channels,
                kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(num_channels))
        self.conv3 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(self.num_convs - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class Kanezaki2018Loss(phmLoss):
    """Loss function implemented based on the loss function defined in,
    @name           Unsupervised Image Segmentation by Backpropagation
    @journal        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    @year           2018
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation
    @citation       Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.
    """

    def __init__(self,
                 compactness: int = 100,
                 superpixel_regions: int = 30
                 ) -> None:
        super().__init__()
        self.compactness = compactness
        self.superpixel_regions = superpixel_regions
        self.l_inds = None
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def prepare_loss(self, **kwargs):
        """Set the reference image for SLIC algorithm duing initialization.

        Args:
            ref (Image): Reference image
        """
        ref = kwargs['ref']
        self._ref = ref
        img_w = ref.shape[0]
        img_h = ref.shape[1]
        # SLIC : segment the image using SLIC algorithm
        labels = segmentation.slic(ref,
            compactness=self.compactness,
            n_segments=self.superpixel_regions)
        # Flatten the resulted segmentation using SLIC
        labels = labels.reshape(img_w * img_h)
        # Extract the unique label
        u_labels = np.unique(labels)
        # Form the label indexes
        self.l_inds = []
        for i in range(len(u_labels)):
            self.l_inds.append(np.where(labels == u_labels[i])[0])

    def forward(self, output, target, **kwargs):
        # Superpixel Refinement
        im_target = target.data.cpu().numpy()
        for i in range(len(self.l_inds)):
            labels_per_sp = im_target[self.l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[self.l_inds[i]] = u_labels_per_sp[np.argmax(hist)]

        return self.loss_fn(output, target)

def create_noref_predict_Kanezaki2018__(
    config_file : str = 'configs/kanezaki2018.json', 
    experiment : Experiment = None,
    metrics : Dict[str,Metric] = None):

    config = load_config(config_file)
    # Initialize model
    model = Kanezaki2018Module(num_dim=3, 
        num_channels=config.model.num_channels, 
        num_convs=config.model.num_conv_layers)
    # Initialize loss
    loss = Kanezaki2018Loss(config.segmentation.compactness,
        config.segmentation.superpixel_regions)
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), 
        lr=config.segmentation.learning_rate, 
        momentum=config.segmentation.momentum)

    if experiment is not None:
        experiment.log_parameters(config.model, prefix='model')
        experiment.log_parameters(config.segmentation, prefix='segmentation')

    seg_obj = KanezakiIterativeSegmentor(
        model=model, 
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        experiment=experiment
    )

    pred_func = functools.partial(
        seg_obj.segment_noref_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics    
    )
    engine = Engine(pred_func)

    if metrics is not None:
        for x in metrics.keys():
            metrics[x].attach(engine, x)

    return engine
