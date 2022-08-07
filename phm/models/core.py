
""" 
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""
import torch
import logging
from typing import Any, Dict, List, Union
import torch.nn as nn
import torch.nn.functional as F

__model_handler = {}

def model_selector(name : Union[str, List[str]]):
    def __embed_func(func):
        global __model_handler
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __model_handler[n] = func

    return __embed_func

def list_models() -> List[str]:
    global __model_handler
    return list(__model_handler.keys())

def load_model(model_name : str, device : str, config):
    if not model_name in list_models():
        msg = f'{model_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __model_handler[model_name](device, **config)

class BaseModule(nn.Module):
    def __init__(self, device : str, config : Dict[str,Any]) -> None:
        super().__init__(
            device=device,
            config=config
        )

# @model_selector('test')
# class Test(BaseModule):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)
