
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

import torch
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import torch.nn.functional as F

from phm.loss import UnsupervisedLoss_TwoFactors
from phm.metrics import phm_Metric
from phm.segment import KanezakiIterativeSegmentor, ignite_segmenter, phmIterativeSegmentor, phmLoss

class Wonjik2020Module (nn.Module):
    """ Implementation of the model presented in:
    @name           Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering   
    @journal        IEEE Transactions on Image Processing
    @year           2020
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
    @citation       Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    """

    def __init__(self, num_dim, num_channels: int = 100, num_convs: int = 3):
        super(Wonjik2020Module, self).__init__()

        self.num_convs = num_convs
        self.num_channel = num_channels

        # First convolutional layer
        self.conv1 = nn.Conv2d(num_dim, self.num_channel,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channel)
        # Feature space including multiple convolutional layers
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        # The number of convolutional layers are determined based on "nCov" parameters.
        for i in range(self.num_convs-1):
            tmpConv = nn.Conv2d(
                self.num_channel, self.num_channel,
                kernel_size=3, stride=1, padding=1)
            tmpBatch = nn.BatchNorm2d(self.num_channel)
            self.conv2.append(tmpConv)
            self.bn2.append(tmpBatch)
        # The reference normalization for extracting class labels
        self.conv3 = nn.Conv2d(
            self.num_channel, self.num_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.num_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(self.num_convs-1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

@ignite_segmenter('wonjik2020')
def generate_wonjik2020_ignite__(
        name: str,
        config: DotMap,
        experiment: Experiment,
        metrics: List[phm_Metric] = None,
        step_metrics: List[phm_Metric] = None,
        category: Dict[str, int] = None,
        **kwargs):

    # Initialize model
    model = Wonjik2020Module(num_dim=3,
                             num_channels=config.model.num_channels,
                             num_convs=config.model.num_conv_layers)
    # Initialize loss
    loss = UnsupervisedLoss_TwoFactors(
        num_channel=config.model.num_channels,
        similarity_loss=config.segmentation.similarity_loss,
        continuity_loss=config.segmentation.continuity_loss
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

@ignite_segmenter('wonjik2020_phm')
def generate_wonjik2020_ignite__(
        name: str,
        config: DotMap,
        experiment: Experiment,
        metrics: List[phm_Metric] = None,
        step_metrics: List[phm_Metric] = None,
        category: Dict[str, int] = None,
        **kwargs):

    # Initialize model
    model = Wonjik2020Module(num_dim=3,
                             num_channels=config.model.num_channels,
                             num_convs=config.model.num_conv_layers)
    # Initialize loss
    loss = UnsupervisedLoss_TwoFactors(
        num_channel=config.model.num_channels,
        similarity_loss=config.segmentation.similarity_loss,
        continuity_loss=config.segmentation.continuity_loss
    )
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
