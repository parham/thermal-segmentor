
""" 
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University
"""

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

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
                 num_dim: int = 3,
                 num_channels: int = 100,
                 part01_kernel_size: int = 3,
                 part01_stride: int = 1,
                 part01_padding: int = 1,
                 part02_num_layer: int = 3,
                 part02_kernel_size: int = 3,
                 part02_stride: int = 2,
                 part02_padding: int = 1,
                 part02_output_padding: int = 1,
                 part03_kernel_size: int = 3,
                 part03_stride: int = 1,
                 part03_padding: int = 2,
                 part04_kernel_size: int = 1,
                 part04_stride: int = 1,
                 part04_padding: int = 0,
                 num_conv_layers: int = 3

                 ):
        super(phmAutoencoderModule, self).__init__()
        # Set the model's config based on provided configuration
        self.nChannel = num_channels
        # Part 01 : the feature extraction
        self.part01 = nn.ModuleList([
            Feature(num_dim, self.nChannel,
                    kernel_size=part01_kernel_size,
                    stride=part01_stride,
                    padding=part01_padding),
            Feature(self.nChannel, self.nChannel,
                    kernel_size=part01_kernel_size,
                    stride=part01_stride,
                    padding=part01_padding),
            Feature(self.nChannel, self.nChannel,
                    kernel_size=part01_kernel_size,
                    stride=part01_stride,
                    padding=part01_padding),
        ])
        # Feature space including multiple convolutional layers
        # Part 02 : Auto-Encoder
        self.encoder = []
        inc_ch = self.nChannel
        # Encoder
        for i in range(part02_num_layer):
            tmp = Feature(inc_ch, inc_ch * 2,
                          kernel_size=part02_kernel_size,
                          stride=part02_stride,
                          padding=part02_padding)
            inc_ch *= 2
            self.encoder.append(tmp)
        self.encoder = nn.ModuleList(self.encoder)
        # Decoder
        self.decoder = []
        for i in range(part02_num_layer):
            tmp = TransposeFeature(int(inc_ch), int(inc_ch / 2),
                                   kernel_size=part02_kernel_size,
                                   stride=part02_stride,
                                   padding=part02_padding,
                                   output_padding=part02_output_padding)
            inc_ch /= 2
            self.decoder.append(tmp)
        self.decoder = nn.ModuleList(self.decoder)
        # Part 03 : the reference normalization for extracting class labels
        self.part03 = nn.ModuleList([
            Feature(self.nChannel, self.nChannel,
                    kernel_size=part03_kernel_size,
                    stride=part03_stride,
                    padding=part03_padding),
            Feature(self.nChannel, self.nChannel,
                    kernel_size=part03_kernel_size,
                    stride=part03_stride,
                    padding=part03_padding),
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
                                   kernel_size=part04_kernel_size,
                                   stride=part04_stride,
                                   padding=part04_padding)

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
                x = torch.cat((x, tmp), dim=-1)
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

