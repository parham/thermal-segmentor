
""" 
    @author        Parham Nooralishahi
    @email         parham.nooralishahi@gmail.com
    @professor     Professor Xavier Maldague
    @organization Laval University
    @description Implementation of a W-Net CNN for unsupervised learning of image segmentations.
                 adopted from https://github.com/fkodom/wnet-unsupervised-image-segmentation
"""

import functools
import numpy as np
from time import time
from typing import Dict, Tuple
from comet_ml import Experiment

from ignite.engine import Engine

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Metric

from phm.core import load_config
from phm.loss import NCutLoss2D, OpeningLoss2D


class ConvPoolBlock(nn.Module):
    """Performs multiple 2D convolutions, followed by a 2D max-pool operation.  Many of these are contained within
    each UNet module, for down sampling image data."""

    def __init__(self, in_features: int, out_features: int):
        """
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(ConvPoolBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_features, out_features, 5),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)

class DeconvBlock(nn.Module):
    """
    Performs multiple 2D transposed convolutions, with a stride of 2 on the last layer.  Many of these are contained
    within each UNet module, for up sampling image data.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(DeconvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ConvTranspose2d(in_features, out_features, 5, padding=2),
            nn.ConvTranspose2d(out_features, out_features, 2, stride=2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)

class OutputBlock(nn.Module):
    """
    Performs multiple 2D convolutions, without any pooling or strided operations.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(OutputBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_features, out_features, 3),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            nn.Conv2d(out_features, out_features, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)

class UNetEncoder(nn.Module):
    """
    The first half (encoder) of the W-Net architecture.  
    Returns class probabilities for each pixel in the image.
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 10):
        """
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super(UNetEncoder, self).__init__()
        self.conv1 = ConvPoolBlock(num_channels, 32)
        self.conv2 = ConvPoolBlock(32, 32)
        self.conv3 = ConvPoolBlock(32, 64)
        self.deconv1 = DeconvBlock(64, 32)
        self.deconv2 = DeconvBlock(64, 32)
        self.deconv3 = DeconvBlock(64, 32)
        self.output = OutputBlock(32, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        x = self.conv3(c2)
        x = self.deconv1(x)
        x = self.deconv2(torch.cat((c2, x), dim=1))
        x = self.deconv3(torch.cat((c1, x), dim=1))
        x = self.output(x)

        return x

class UNetDecoder(nn.Module):
    """
    The second half (decoder) of the W-Net architecture.  
    Returns a reconstruction of the original image.
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 20):
        """
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super(UNetDecoder, self).__init__()
        self.conv1 = ConvPoolBlock(num_classes, 32)
        self.conv2 = ConvPoolBlock(32, 32)
        self.conv3 = ConvPoolBlock(32, 64)
        self.deconv1 = DeconvBlock(64, 32)
        self.deconv2 = DeconvBlock(64, 32)
        self.deconv3 = DeconvBlock(64, 32)
        self.output = OutputBlock(32, num_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        x = self.conv3(c2)
        x = self.deconv1(x)
        x = self.deconv2(torch.cat((c2, x), dim=1))
        x = self.deconv3(torch.cat((c1, x), dim=1))
        x = self.output(x)

        return x

class WNet(nn.Module):
    """
    Implements a W-Net CNN model for learning unsupervised image segmentations.  First encodes image data into
    class probabilities using UNet, and then decodes the labels into a reconstruction of the original image using a
    second UNet.
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 20):
        """
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super(WNet, self).__init__()
        self.encoder = UNetEncoder(num_channels=num_channels, num_classes=num_classes)
        self.decoder = UNetDecoder(num_channels=num_channels, num_classes=num_classes)

    def forward_encode_(self, x: Tensor) -> Tensor:
        """
        Pushes a set of inputs (x) through only the encoder network.

        :param x: Input values
        :return: Class probabilities
        """

        return self.encoder(x)


        return loss

class WNet(nn.Module):
    """
    Implements a W-Net CNN model for learning unsupervised image segmentations.  
    First encodes image data into class probabilities using UNet, and then decodes 
    the labels into a reconstruction of the original image using a second UNet.
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 20):
        """
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super(WNet, self).__init__()
        self.encoder = UNetEncoder(num_channels=num_channels, num_classes=num_classes)
        self.decoder = UNetDecoder(num_channels=num_channels, num_classes=num_classes)

    def forward_encode_(self, x: Tensor) -> Tensor:
        """
        Pushes a set of inputs (x) through only the encoder network.

        :param x: Input values
        :return: Class probabilities
        """

        return self.encoder(x)

    def forward_reconstruct_(self, mask: Tensor) -> Tensor:
        """
        Pushes a set of class probabilities (mask) through only the decoder network.

        :param mask: Class probabilities
        :return: Reconstructed image
        """

        outputs = self.decoder(mask)
        outputs = nn.ReLU()(outputs)

        return outputs

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """

        encoded = self.forward_encode_(x).transpose(1, -1)
        mask = nn.Softmax(-1)(encoded).transpose(-1, 1)
        reconstructed = self.forward_reconstruct_(mask)

        return mask, reconstructed

class WNetLoss(nn.Module):
    def __init__(self, 
        alpha = 1e-3, 
        beta = 1, 
        gamma = 1e-1
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, output, target, input, mask): # labels > target
        input, label, output = input.contiguous(), target.contiguous(), output.contiguous()
        # Weights for NCutLoss2D, MSELoss, and OpeningLoss2D, respectively
        ncut_loss = self.alpha * NCutLoss2D()(mask, input)
        mse_loss = self.beta * nn.MSELoss()(output, input.detach())
        smooth_loss = self.gamma * OpeningLoss2D()(mask)
        loss = ncut_loss + mse_loss + smooth_loss

class WNet_Impl:

    def __init__(self, 
        model,
        optimizer,
        loss,
        experiment: Experiment = None
    ) -> None:
        super().__init__(experiment)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss

    def fit_step__(self, engine, batch, 
        log_img : bool = True, 
        log_metrics : bool = True):
        return self.fit(batch[0], batch[0], batch[1], 
            log_img = log_img, 
            log_metrics = log_metrics)

    def fit(self, img, target,
        log_img : bool = True, 
        log_metrics : bool = True):

        if log_img:
            self.experiment.log_image(
                img, name='original', step=0)

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        mask, output = self.forward(img)
        loss = self.loss_fn(output, target, img, mask)
        loss.backward()
        self.optimizer.step()
        step_time = time.time() - t

        _, target_ = torch.max(output, 1)
        im_target = target_.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        seg_result = im_target.reshape(img.shape[0:2])

        if log_metrics:
            self.experiment.log_metrics({
                'noref_step_time': step_time,
                'noref_class_count': nLabels,
                'noref_loss': loss
            }, step=1, epoch=1)
        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=1)
