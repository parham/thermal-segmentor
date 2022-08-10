
""" 
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

from typing import Any, Dict
from phm.models.core import BaseModule, model_selector
import segmentation_models_pytorch as smp

# @model_selector('unet_resnet18')
class Unet_Resnet18(BaseModule):
    def __init__(self, device : str, config : Dict[str,Any]) -> None:
        super().__init__(
            device=device,
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

# @model_selector('unetplusplus_resnet18')
class UnetPlusPlus_Resnet18(BaseModule):
    def __init__(self, device : str, config : Dict[str,Any]) -> None:
        super().__init__(
            device=device,
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

# @model_selector('fpn_resnet18')
class FPN_Resnet18(BaseModule):
    def __init__(self, device : str, config : Dict[str,Any]) -> None:
        super().__init__(
            device=device,
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