
import torch
import logging
import numpy as np

from typing import Any, Callable, Dict, List, Union
from dataclasses import dataclass
from comet_ml import Experiment
from ignite.engine import Engine

label_colors_1ch8bits = np.random.randint(10,255,size=(100,1))

__segmenter_handler = {}

def segmenter_method(name : Union[str, List[str]]):
    def __embed_func(func):
        global __segmenter_handler
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __segmenter_handler[n] = func

    return __embed_func

def list_segmenter_methods() -> List[str]:
    global __segmenter_handler
    return list(__segmenter_handler.keys())

def segment_loader(
    handler : str,
    data_loader,
    category : Dict,
    experiment : Experiment,
    config = None,
    device : str = None
) -> Engine:
    device_non = device if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not handler in __segmenter_handler.keys():
        msg = f'{handler} handler is not supported!'
        logging.error(msg)
        raise ValueError(msg)

    experiment.log_parameters(config.model)
    experiment.log_parameters(config.segmentation)
    experiment.log_parameters(config.general)

    return __segmenter_handler[handler](
        handler=handler,
        category=category,
        experiment=experiment,
        config=config,
        device=device_non
    )

def simplify_train_step(
    experiment : Experiment,
    call_segment_func : Callable
):
    def __train_step(engine, batch):
        # Log the image and target
        if engine.state.log_image:
            target = batch[1] if len(batch) > 1 else None
            img = batch[0].cpu().numpy()
            img = np.squeeze(img, axis=0)
            
            experiment.log_image(img, 
                overwrite=True,
                name=f'original', 
                step=engine.state.iteration)
            
            if target is not None:
                experiment.log_image(target, 
                    overwrite=True,
                    name=f'target', 
                    step=engine.state.iteration)

        # Recall the step
        res = call_segment_func(engine, batch)

        if engine.state.log_metrics:
            if res.internal_metrics is not None and res.internal_metrics:
                experiment.log_metrics(res.internal_metrics, prefix='loop_',
                    step=engine.state.iteration, epoch=engine.state.epoch)

        if engine.state.log_image:
            experiment.log_image(res.output, 
                name=f'adapted_result', 
                step=engine.state.iteration)
            experiment.log_image(res.output_ready, 
                name=f'result', 
                step=engine.state.iteration)
    
    return __train_step

@dataclass
class SegmentRecord:
    loss : float
    output : Any
    target : Any = None
    output_ready : Any = None
    internal_metrics : Dict = None

class GrayToRGB(torch.nn.Module):
    def forward(self, sample) -> Any:
        res = sample
        if len(sample.shape) < 3:
            res = np.expand_dims(res, axis=2)
            res = np.concatenate((res,res,res), axis=2)
        return res