
""" 
    @author        Parham Nooralishahi
    @email         parham.nooralishahi@gmail.com
    @professor     Professor Xavier Maldague
    @organization: Laval University
"""

import functools
import logging
from typing import Dict, Tuple
from comet_ml import Experiment
from dotmap import DotMap
import torch
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import torch.nn.functional as F

from torchmetrics import Metric
from ignite.engine import Engine

import time
import numpy as np

from phm.core import load_config


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


class Wonjik2020Loss(nn.Module):
    """ Loss function implemented based on the loss function defined in,
    @name           Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering   
    @journal        IEEE Transactions on Image Processing
    @year           2020
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
    @citation       Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    """

    def __init__(self,
                 num_channel: int = 100,
                 similarity_loss: float = 1.0,
                 continuity_loss: float = 0.5
                 ) -> None:
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_hpy = torch.nn.L1Loss(size_average=True)
        self.loss_hpz = torch.nn.L1Loss(size_average=True)
        self.HPy_target = None
        self.HPz_target = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_loss = similarity_loss
        self.continuity_loss = continuity_loss
        self.nChannel = num_channel

    def set_ref(self, ref):
        self._ref = ref
        img_w = ref.shape[0]
        img_h = ref.shape[1]
        self.HPy_target = torch.zeros(
            img_w - 1, img_h, self.nChannel).to(self.device)
        self.HPz_target = torch.zeros(
            img_w, img_h - 1, self.nChannel).to(self.device)

    def forward(self, output, target, img_size: Tuple[int, int, int]):
        img_w = img_size[0]
        img_h = img_size[1]

        outputHP = output.reshape((img_w, img_h, self.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = self.loss_hpy(HPy, self.HPy_target)
        lhpz = self.loss_hpz(HPz, self.HPz_target)
        # loss calculation
        return self.similarity_loss * self.loss_fn(output, target) + \
            self.continuity_loss * (lhpy + lhpz)


class Wonjik2020_Impl:
    """ Implementation of unsupervised segmentation method presented in,
    @name           Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering   
    @journal        IEEE Transactions on Image Processing
    @year           2020
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
    @citation       Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    """

    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 num_channel: int = 100,
                 iteration: int = 100,
                 min_classes: int = 10,
                 experiment: Experiment = None) -> None:

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss
        self.experiment = experiment
        # Number of channels
        self.nChannel = num_channel
        self.iteration = iteration
        self.min_classes = min_classes
        self.last_label_count = 0

    def unsupervise_segmentation_step__(self, engine, batch, log_img : bool = True, log_metrics : bool = True):
        img = batch[0]
        self.last_label_count = 0
        return self.unsupervise_segmentation(img, log_img = log_img, log_metrics = log_metrics)

    def unsupervise_segmentation(self, img,
                                 log_img: bool = True,
                                 log_metrics: bool = True):

        last_loss = None
        seg_result = None
        seg_step_time = 0

        img_w = img.shape[0]
        img_h = img.shape[1]
        # Convert image to numpy data
        img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])

        self.experiment.log_image(
            img, name='original', step=0)
        # Convert image to tensor
        data = torch.from_numpy(img_data).to(self.device)

        self.loss_fn.set_ref(img)

        self.model.train()
        with self.experiment.train():
            for step in range(self.iteration):
                t = time.time()
                self.optimizer.zero_grad()
                output = self.model(data)[0, :, 0:img_w, 0:img_h]

                output_orig = output.permute(1, 2, 0).contiguous()
                output = output_orig.view(-1, self.nChannel)

                _, target = torch.max(output, 1)
                im_target = target.data.cpu().numpy()
                nLabels = len(np.unique(im_target))

                seg_result = im_target.reshape(img.shape[0:2])

                target = torch.from_numpy(im_target).to(self.device)

                loss = self.loss_fn(output, target, img.shape)
                loss.backward()
                last_loss = loss
                self.optimizer.step()

                logging.info(
                    f'{step} / {self.iteration} : {nLabels} , {loss.item()}')

                step_time = time.time() - t
                seg_step_time += step_time
                self.last_label_count = nLabels

                if log_metrics:
                    self.experiment.log_metrics({
                        'noref_step_time': step_time,
                        'noref_class_count': nLabels,
                        'noref_loss': loss
                    }, step=step, epoch=1)
                if log_img:
                    self.experiment.log_image(
                        seg_result, name='steps', step=step)

                if nLabels <= self.min_classes:
                    logging.info(
                        f'Number of labels has reached {self.last_label_count}.')
                    break

        return last_loss.item(), seg_result

def create_noref_predict_Wonjik2020__(
    config_file : str = 'configs/wonjik2020.json', 
    experiment : Experiment = None,
    metrics : Dict[str,Metric] = None):

    config = load_config(config_file)
    # Initialize model
    model = Wonjik2020Module(num_dim=3, 
        num_channels=config.model.num_channels, 
        num_convs=config.model.num_conv_layers)
    # Initialize loss
    loss = Wonjik2020Loss(
        num_channel = config.model.num_channels,
        similarity_loss = config.segmentation.similarity_loss,
        continuity_loss = config.segmentation.continuity_loss
    )
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), 
        lr=config.segmentation.learning_rate, 
        momentum=config.segmentation.momentum)

    if experiment is not None:
        experiment.log_parameters(config.model, prefix='model')
        experiment.log_parameters(config.segmentation, prefix='segmentation')

    seg_obj = Wonjik2020_Impl(
        model=model, 
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        experiment=experiment
    )

    pred_func = functools.partial(seg_obj.unsupervise_segmentation_step__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics    
    )
    engine = Engine(pred_func)

    if metrics is not None:
        for x in metrics.keys():
            metrics[x].attach(engine, x)

    return engine