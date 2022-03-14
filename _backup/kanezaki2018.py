
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from skimage import segmentation

from typing import Any, Dict
from pytorch_lightning import LightningModule
from pytorch_lightning.loops import Loop
from dotmap import DotMap

class Kanezaki2018Loop (Loop):

    def __init__(self, model, optimizer, dataloader, config):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.batch_idx = 0
        self.nClasses = config.segmentation.min_classes
        self.maxLoop = config.segmentation.iteration

    @property
    def done(self):
        return self.batch_idx >= len(self.dataloader)

    def reset(self) -> None:
        self.dataloader_iter = iter(self.dataloader)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        batch = next(self.dataloader_iter)
        nLabel = self.nClasses + 10
        # for loop_count in range(self.maxLoop):
        # if nLabel <= self.nClasses:
        #     break
        self.optimizer.zero_grad()
        metrics = self.model.training_step(batch, self.batch_idx)
        nLabel = metrics['num_label']
        loss = metrics['loss']
        loss.backward()
        self.optimizer.step()
        

class Kanezaki2018Module (LightningModule):

    def __init__(self, numDim : int, configs : Dict = {}, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.configs = configs if type(configs) != DotMap else DotMap(configs)
        self.nDim = numDim
        self.__model()
    
    def __model(self) :
        nChannel = self.configs.model.num_channels
        nConv = self.configs.model.num_conv_layers
        nDim = self.nDim

        self.conv1 = nn.Conv2d(
            nDim, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        # Form internal neural structure
        for i in range(nConv-1):
            self.conv2.append(nn.Conv2d(nChannel, nChannel,
                kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannel))
        self.conv3 = nn.Conv2d(
            nChannel, nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        nConv = self.configs.model.num_conv_layers

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

    def configure_optimizers(self):
        lr = self.configs.segmentation.learning_rate
        momentum = self.configs.segmentation.momentum
        return optim.SGD(self.parameters(), lr=lr, momentum=momentum)

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

    def training_step(self, batch, batch_idx):
        img = batch[0]
        img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])
        data = torch.from_numpy(img_data)
        if self.use_cuda:
            data = data.cuda()
        data = Variable(data)

        img_w = img.shape[0]
        img_h = img.shape[1]

        nChannel = self.configs.model.num_channels
        output = self.model(data)[0,:,0:img_w,0:img_h]
        output_orig = output.permute(1, 2, 0).contiguous()
        output = output_orig.view(-1, nChannel)

        _, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        target = torch.from_numpy(im_target)
        if self.use_cuda:
            target = target.cuda()

        target = Variable(target)
        loss = self.calc_loss(output, target)

        return {
            'loss' : loss,
            'num_label' : nLabels
        }

