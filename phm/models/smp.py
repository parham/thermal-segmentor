
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

from dotmap import DotMap
from phm.models.core import BaseModule, model_register
import segmentation_models_pytorch as smp

@model_register('unet_resnet18')
class Unet_Resnet18(BaseModule):
    def __init__(self, name : str, config : DotMap) -> None:
        super().__init__(
            name='unet_resnet18',
            config=config
        )
        self.clss = smp.Unet(
            encoder_name='resnet18',
            encoder_weights="imagenet",
            in_channels=self.channels,
            classes=self.num_classes 
        )
    
    def forward(self, x):
        return self.clss(x)

@model_register('unetplusplus_resnet18')
class UnetPlusPlus_Resnet18(BaseModule):
    def __init__(self, name : str, config : DotMap) -> None:
        super().__init__(
            name='unetplusplus_resnet18',
            config=config
        )
        self.clss = smp.UnetPlusPlus(
            encoder_name='resnet18',
            encoder_weights="imagenet",
            in_channels=self.channels,
            classes=self.num_classes 
        )

    def forward(self, x):
        return self.clss(x)

@model_register('fpn_resnet18')
class FPN_Resnet18(BaseModule):
    def __init__(self, name : str, config : DotMap) -> None:
        super().__init__(
            name='fpn_resnet18',
            config=config
        )
        self.clss = smp.FPN(
            encoder_name='resnet18',
            encoder_weights="imagenet",
            in_channels=self.channels,
            classes=self.num_classes 
        )

    def forward(self, x):
        return self.clss(x)