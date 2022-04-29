
""" 
    @author        Parham Nooralishahi
    @email         parham.nooralishahi@gmail.com
    @professor     Professor Xavier Maldague
    @organization: Laval University
"""

import functools
from typing import Dict, List
from comet_ml import Experiment
from dotmap import DotMap

import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import torch.nn.functional as F
from phm.loss import UnsupervisedLoss_SuperResolusion

from torchmetrics import Metric
from ignite.engine import Engine, EventEnum

from phm.core import load_config
from phm.metrics import phm_Metric
from phm.segment import KanezakiIterativeSegmentor, ignite_segmenter, phmIterativeSegmentor

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

@ignite_segmenter('kanezaki2018')
def generate_kanezaki2018_ignite__(
    name : str,
    config : DotMap,
    experiment : Experiment,
    metrics : List[phm_Metric] = None,
    step_metrics : List[phm_Metric] = None,
    category : Dict[str, int] = None,
    **kwargs):

    # Initialize model
    model = Kanezaki2018Module(num_dim=3, 
        num_channels=config.model.num_channels, 
        num_convs=config.model.num_conv_layers)
    # Initialize loss
    loss = UnsupervisedLoss_SuperResolusion(
        config.segmentation.compactness,
        config.segmentation.superpixel_regions
    )
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), 
        lr=config.segmentation.learning_rate, 
        momentum=config.segmentation.momentum)

    seg_obj = KanezakiIterativeSegmentor(
        model=model, 
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        experiment=experiment,
        metrics=metrics,
        step_metrics=step_metrics,
        category=category
    )

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics    
    )
    
    return seg_obj, pred_func

@ignite_segmenter('kanezaki2018_phm')
def generate_kanezaki2018_ignite__(
    name : str,
    config : DotMap,
    experiment : Experiment,
    metrics : List[phm_Metric] = None,
    step_metrics : List[phm_Metric] = None,
    category : Dict[str, int] = None,
    **kwargs):

    # Initialize model
    model = Kanezaki2018Module(num_dim=3, 
        num_channels=config.model.num_channels, 
        num_convs=config.model.num_conv_layers)
    # Initialize loss
    loss = UnsupervisedLoss_SuperResolusion(config.segmentation.compactness,
        config.segmentation.superpixel_regions)
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), 
        lr=config.segmentation.learning_rate, 
        momentum=config.segmentation.momentum)

    seg_obj = phmIterativeSegmentor(
        model=model, 
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        min_area=config.segmentation.min_area,
        experiment=experiment,
        metrics=metrics,
        step_metrics=step_metrics,
        category=category
    )

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics    
    )
    
    return seg_obj, pred_func