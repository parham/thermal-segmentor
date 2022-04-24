
""" 
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University
"""

import functools
import logging
from typing import Dict, Tuple
from dotmap import DotMap

import torch
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Engine

import time
import numpy as np
from torchmetrics import Metric
from comet_ml import Experiment

from phm.core import load_config
from phm.segment import Segmentor, ignite_segmenter, phmLoss

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
        # self.cls = nn.Softmax2d()

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
    def __init__(self, 
        num_dim : int = 3,
        num_channels : int = 100,
        part01_kernel_size : int = 3,
        part01_stride : int = 1,
        part01_padding : int = 1,
        part02_num_layer : int = 3,
        part02_kernel_size : int = 3,
        part02_stride : int = 2,
        part02_padding : int = 1,
        part02_output_padding : int = 1,
        part03_kernel_size : int = 3,
        part03_stride : int = 1,
        part03_padding : int = 2,
        part04_kernel_size : int = 1,
        part04_stride : int = 1,
        part04_padding : int = 0,
        num_conv_layers : int = 3

    ):
        super(phmAutoencoderModule, self).__init__()
        # Set the model's config based on provided configuration
        self.nChannel = num_channels
        # Part 01 : the feature extraction
        self.part01 = nn.ModuleList([
            Feature(num_dim, self.nChannel,
                kernel_size = part01_kernel_size,
                stride = part01_stride,
                padding = part01_padding),
            Feature(self.nChannel, self.nChannel,
                kernel_size = part01_kernel_size,
                stride = part01_stride,
                padding = part01_padding),
            Feature(self.nChannel, self.nChannel,
                kernel_size = part01_kernel_size,
                stride = part01_stride,
                padding = part01_padding),
        ])
        # Feature space including multiple convolutional layers
        # Part 02 : Auto-Encoder
        self.encoder = []
        inc_ch = self.nChannel
        # Encoder
        for i in range(part02_num_layer):
            tmp = Feature(inc_ch, inc_ch * 2,
                kernel_size = part02_kernel_size,
                stride = part02_stride,
                padding = part02_padding)
            inc_ch *= 2
            self.encoder.append(tmp)
        self.encoder = nn.ModuleList(self.encoder)
        # Decoder
        self.decoder = []
        for i in range(part02_num_layer):
            tmp = TransposeFeature(int(inc_ch), int(inc_ch / 2),
                kernel_size = part02_kernel_size,
                stride = part02_stride,
                padding = part02_padding,
                output_padding = part02_output_padding)
            inc_ch /= 2
            self.decoder.append(tmp)
        self.decoder = nn.ModuleList(self.decoder)
        # Part 03 : the reference normalization for extracting class labels
        self.part03 = nn.ModuleList([
            Feature(self.nChannel, self.nChannel,
                kernel_size = part03_kernel_size,
                stride = part03_stride,
                padding = part03_padding),
            Feature(self.nChannel, self.nChannel,
                kernel_size = part03_kernel_size,
                stride = part03_stride,
                padding = part03_padding),
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
            kernel_size = part04_kernel_size,
            stride = part04_stride,
            padding = part04_padding)

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

class phmAutoencoderLoss(phmLoss):
    def __init__(self,
        num_channel: int = 100,
        similarity_loss: float = 0.99,
        continuity_loss: float = 0.5
    ) -> None:
        super().__init__()
        # similarity loss definition
        self.loss_fn = nn.CrossEntropyLoss()
        # continuity loss definition
        self.loss_hpy = torch.nn.SmoothL1Loss(size_average=True)
        self.loss_hpz = torch.nn.SmoothL1Loss(size_average=True)

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
            img_w-1, img_h, self.nChannel).to(self.device)
        self.HPz_target = torch.zeros(
            img_w, img_h-1, self.nChannel).to(self.device)

    def forward(self, output, target, img_size: Tuple):
        img_w = img_size[0]
        img_h = img_size[1]
        outputHP = output.reshape((img_w, img_h, self.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = self.loss_hpy(HPy, self.HPy_target)
        lhpz = self.loss_hpz(HPz, self.HPz_target)
        # loss calculation
        return self.similarity_loss * self.loss_fn(output.view(-1, self.nChannel), target.view(-1)) + \
            self.continuity_loss * (lhpy + lhpz)

class phmAutoencoder_Impl(Segmentor):
    
    def __init__(self, 
        model,
        optimizer,
        loss,
        num_channel: int = 100,
        iteration: int = 100,
        min_classes: int = 10,
        experiment: Experiment = None) -> None:

        super().__init__(experiment)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss
        
        # Number of channels
        self.nChannel = num_channel
        self.iteration = iteration
        self.min_classes = min_classes
        self.last_label_count = 0

    def segment_noref_step__(self, engine, batch,
        log_img: bool = True,
        log_metrics: bool = True):
        img = batch[0]
        self.last_label_count = 0
        return self.segment_noref(img, log_img = log_img, log_metrics = log_metrics)

    def segment_noref(self, img,
        log_img: bool = True,
        log_metrics: bool = True):

        last_loss = None
        seg_result = None
        seg_step_time = 0

        img_w = img.shape[0]
        img_h = img.shape[1]
        # Convert image to numpy data
        img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])

        if log_img:
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
                output = self.model(data)[0,:,0:img_w,0:img_h]

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

@ignite_segmenter('phm_autoencoder')
def generate_phm_autoencoder_ignite__(
    name : str,
    config : DotMap,
    experiment : Experiment):

    # Initialize model
    model = phmAutoencoderModule(num_dim=3, 
        num_channels=config.model.num_channels,
        part01_kernel_size = config.model.part01_kernel_size,
        part01_stride = config.model.part01_stride,
        part01_padding = config.model.part01_padding,
        part02_num_layer = config.model.part02_num_layer,
        part02_kernel_size = config.model.part02_kernel_size,
        part02_stride = config.model.part02_stride,
        part02_padding = config.model.part02_padding,
        part02_output_padding = config.model.part02_output_padding,
        part03_kernel_size = config.model.part03_kernel_size,
        part03_stride = config.model.part03_stride,
        part03_padding = config.model.part03_padding,
        part04_kernel_size = config.model.part04_kernel_size,
        part04_stride = config.model.part04_stride,
        part04_padding = config.model.part04_padding,
        num_conv_layers = config.model.num_conv_layers
    )
    # Initialize loss
    loss = phmAutoencoderLoss(
        num_channel = config.model.num_channels,
        similarity_loss = config.segmentation.similarity_loss,
        continuity_loss = config.segmentation.continuity_loss)
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), 
        lr=config.segmentation.learning_rate, 
        momentum=config.segmentation.momentum)

    seg_obj = phmAutoencoder_Impl(
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

    return seg_obj, pred_func


def create_noref_predict_phmAutoencoder__(
    config_file : str = 'configs/wonjik2020.json', 
    experiment : Experiment = None,
    metrics : Dict[str,Metric] = None):

    config = load_config(config_file)
    # Initialize model
    model = phmAutoencoderModule(num_dim=3, 
        num_channels=config.model.num_channels,
        part01_kernel_size = config.model.part01_kernel_size,
        part01_stride = config.model.part01_stride,
        part01_padding = config.model.part01_padding,
        part02_num_layer = config.model.part02_num_layer,
        part02_kernel_size = config.model.part02_kernel_size,
        part02_stride = config.model.part02_stride,
        part02_padding = config.model.part02_padding,
        part02_output_padding = config.model.part02_output_padding,
        part03_kernel_size = config.model.part03_kernel_size,
        part03_stride = config.model.part03_stride,
        part03_padding = config.model.part03_padding,
        part04_kernel_size = config.model.part04_kernel_size,
        part04_stride = config.model.part04_stride,
        part04_padding = config.model.part04_padding,
        num_conv_layers = config.model.num_conv_layers
    )
    # Initialize loss
    loss = phmAutoencoderLoss(
        num_channel = config.model.num_channels,
        similarity_loss = config.segmentation.similarity_loss,
        continuity_loss = config.segmentation.continuity_loss)
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), 
        lr=config.segmentation.learning_rate, 
        momentum=config.segmentation.momentum)

    if experiment is not None:
        experiment.log_parameters(config.model, prefix='model')
        experiment.log_parameters(config.segmentation, prefix='segmentation')

    seg_obj = phmAutoencoder_Impl(
        model=model, 
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        experiment=experiment
    )

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics    
    )
    engine = Engine(pred_func)

    if metrics is not None:
        for x in metrics.keys():
            metrics[x].attach(engine, x)

    return engine