
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

from xmlrpc.client import boolean
import torch
import torch.nn as nn
import torch.nn.functional as F

from lemanchot.models.core import BaseModule, model_register

class phmBlock(nn.Module):
    def __init__(self,
        in_channels : int,
        out_channels : int,
        kernel_size : int,
        stride : int,
        padding : int,
        zero_probability : float,
        dropout_enable : boolean = True,
        relu_enable : boolean = True             
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.dp = nn.Dropout2d(zero_probability)
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.dropout_enable = dropout_enable
        self.relu_enable = relu_enable
    
    def forward(self, out):
        x = self.conv(out)
        if self.relu_enable:
            x = F.relu(x)
        if self.dropout_enable:
            x = self.dp(x)
        x = self.bn(x)
        return x

@model_register('phm_iterative')
class phmIterativeModule (BaseModule):

    def __init__(self, name : str, config) -> None:
        super().__init__(
            name='phm_iterative',
            config=config
        )
        # Bock #1
        self.block1 = phmBlock(
            in_channels=self.num_dim,
            out_channels=self.num_channels,
            kernel_size=3, 
            stride=1, 
            padding=1,
            zero_probability=self.zero_probability
        )
        # Block #2
        self.block2 = phmBlock(
            in_channels=self.num_channels + self.num_dim,
            out_channels=self.num_channels,
            kernel_size=3, 
            stride=1, 
            padding=1,
            zero_probability=self.zero_probability
        )
        # Block List : the number of convolutional layers are 
        #              determined based on "nCov" parameters.
        self.block_list = nn.ModuleList()
        for i in range(self.num_convs-1):
            self.block3 = phmBlock(
                in_channels=self.num_channels * 2,
                out_channels=self.num_channels,
                kernel_size=3, 
                stride=1, 
                padding=1,
                zero_probability=self.zero_probability
            )
            self.block_list.append(self.block3)
        # Block #N+1 the reference normalization for extracting class labels
        self.blockN = phmBlock(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=1, 
            stride=1, 
            padding=0,
            zero_probability=self.zero_probability,
            relu_enable=False
        )

    def forward(self, out):
        x1 = self.block1(out) # x1 : num_channels
        out1 = torch.cat([x1, out], 1) # out1 : num_channels + num_dim
        x2 = self.block2(out1) # x2 : num_channels
        outLoop = torch.cat([x2, x1], 1) # out2 : num_channels * 2
        xLoop = x2
        for i in range(self.num_convs-1):
            x3 = self.block3(outLoop) # x3 : num_channels
            outLoop = torch.cat([x3, xLoop], 1) # out2 : num_channels * 2
            xLoop = x3
        xN = self.blockN(xLoop)
        return xN
