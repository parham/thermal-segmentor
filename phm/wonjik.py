
""" 
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University

    @name           Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering   
    @journal        IEEE Transactions on Image Processing
    @year           2020
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
    @citation       Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    @info           the code is based on the implementation presented in the mentioned repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

from phm import KWIterativeNNSegmentator


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


class Wonjik2020Segmentator(KWIterativeNNSegmentator):

    def __init__(self,
        seg_config,
        model=None,
        optimizer=None,
    ) -> None:
        super().__init__(seg_config=seg_config, model=model, optimizer=optimizer)
        # similarity loss definition
        self.loss_fn = nn.CrossEntropyLoss()
        # continuity loss definition
        self.loss_hpy = torch.nn.L1Loss(size_average=True)
        self.loss_hpz = torch.nn.L1Loss(size_average=True)
        self.HPy_target = None
        self.HPz_target = None

    def pre_segment(self, img):
        self.HPy_target = torch.zeros(
            img.shape[0]-1, img.shape[1], self.nChannel)
        self.HPz_target = torch.zeros(
            img.shape[0], img.shape[1]-1, self.nChannel)
        if self.use_cuda:
            self.HPy_target = self.HPy_target.cuda()
            self.HPz_target = self.HPz_target.cuda()

        if self.model is None:
            self.model = Wonjik2020Net(self.model_config, img.shape[2])

        super().pre_segment(img)

    def calc_loss(self, img, output, target):
        outputHP = output.reshape((img.shape[0], img.shape[1], self.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = self.loss_hpy(HPy, self.HPy_target)
        lhpz = self.loss_hpz(HPz, self.HPz_target)
        # loss calculation
        return self.similarity_loss_ssize * self.loss_fn(output, target) + self.continuity_loss_ssize * (lhpy + lhpz)
