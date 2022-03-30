
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

from phm.core import Segmentor, load_config

class Kanezaki2018Events(EventEnum):
    INTERNAL_TRAIN_LOOP_COMPLETED = 'internal_train_loop_completed'

class Kanezaki2018Module (nn.Module):
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

class Kanezaki2018Loss(nn.Module):
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

    def set_ref(self, ref):
        """Set the reference image for SLIC algorithm duing initialization.

        Args:
            ref (Image): Reference image
        """
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

    def forward(self, output, target, ref=None):
        if ref is not None:
            self.set_ref(ref)
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

class Kanezaki2018_Impl(Segmentor):
    """Implementation of unsupervised segmentation method presented in,
    @name           Unsupervised Image Segmentation by Backpropagation
    @journal        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    @year           2018
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation
    @citation       Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.
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

    def check_label_limit(self, engine : Engine):
        if self.last_label_count <= self.min_classes:
            logging.info(f'Number of labels has reached {self.last_label_count}.')
            engine.terminate()

    def segment_noref_step__(self, engine, batch, 
        log_img : bool = True, 
        log_metrics : bool = True):
        img = batch[0]
        self.last_label_count = 0
        return self.segment_noref(img, log_img = log_img, log_metrics = log_metrics)

    def segment_noref(self, img,
        log_img : bool = True, 
        log_metrics : bool = True):
        """Segment an image using the unsupervise approach presented in,
        Unsupervised Image Segmentation by Backpropagation.

        Args:
            trainer (Engine): Pytorch ignite Engine object
            batch (array of images): the input data
        """

        last_loss = None
        seg_result = None
        seg_step_time = 0

        img_w = img.shape[0]
        img_h = img.shape[1]
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

                loss = self.loss_fn(output, target)
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
                    logging.info(f'Number of labels has reached {self.last_label_count}.')
                    break

        return last_loss.item(), seg_result


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

    seg_obj = Kanezaki2018_Impl(
        model=model, 
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        experiment=experiment
    )

    pred_func = functools.partial(
        seg_obj.segment_noref_step__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics    
    )
    engine = Engine(pred_func)

    if metrics is not None:
        for x in metrics.keys():
            metrics[x].attach(engine, x)

    return engine
