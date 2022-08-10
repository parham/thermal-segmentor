
""" 
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""
import logging
from typing import Any, Dict, List, Union

from phm.core import phmCore

__model_handler = {}

def model_selector(name : Union[str, List[str]]):
    def __embed_func(clss):
        global __model_handler
        hname = name if isinstance(name, list) else [name]
        if not issubclass(clss, BaseModule):
            raise NotImplementedError('The specified model is not correctly implemented!')
        for n in hname:
            __model_handler[n] = clss

    return __embed_func

def list_models() -> List[str]:
    global __model_handler
    return list(__model_handler.keys())

def load_model(model_name : str, device : str, config):
    if not model_name in list_models():
        msg = f'{model_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __model_handler[model_name](model_name, device, config)

class BaseModule(phmCore):
    def __init__(self, name : str, device : str, config : Dict[str,Any]) -> None:
        super().__init__(
            name=name,
            device=device,
            config=config
        )

# @model_selector('test')
# class Test(BaseModule):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)
