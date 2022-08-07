

""" 
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import logging
from typing import Any, Dict, List, Union

from phm.core import phmCore

class BaseLoss(phmCore):
    def __init__(self, device : str, config : Dict[str,Any]) -> None:
        super().__init__(
            device=device,
            config=config
        )
        
    def prepare_loss(self, **kwargs):
        return

    def forward(self, output, target, **kwargs):
        super().forward(output, target, **kwargs)

__loss_handler = {}

def loss_selector(name : Union[str, List[str]]):
    def __embed_func(clss):
        global __loss_handler
        hname = name if isinstance(name, list) else [name]
        if not isinstance(clss, BaseLoss):
            raise NotImplementedError('The specified loss handler is not implemented!')
        for n in hname:
            __loss_handler[n] = clss

    return __embed_func

def list_losses() -> List[str]:
    global __loss_handler
    return list(__loss_handler.keys())

def load_loss(loss_name : str, device : str, config : Dict[str,Any]):
    if not loss_name in list_losses():
        msg = f'{loss_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __loss_handler[loss_name](device, config)

@loss_selector('neutral')
class NeutralLoss(BaseLoss):
    def __init__(self, 
        device: str, 
        config: Dict[str, Any]
    ) -> None:
        super().__init__(device, config)