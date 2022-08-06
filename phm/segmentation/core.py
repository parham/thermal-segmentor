
import torch
import torchvision.transforms as T

import logging
import numpy as np

from typing import Any, Callable, Dict, List, Union
from dataclasses import dataclass
from comet_ml import Experiment
from ignite.engine import Engine
from phm.loss.core import phmLoss

from phm.metrics import phm_Metric
from phm.models.core import BaseModule
from phm.postprocessing import adapt_output, remove_small_regions

label_colors_1ch8bits = np.random.randint(10,255,size=(100,1))

__segmenter_handler = {}

@dataclass
class SegmentRecord:
    loss : float
    output : Any
    target : Any = None
    output_ready : Any = None
    internal_metrics : Dict = None

def segmenter_method(name : Union[str, List[str]]):
    def __embed_func(func):
        global __segmenter_handler
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __segmenter_handler[n] = func

    return __embed_func

def list_segmenters() -> List[str]:
    global __segmenter_handler
    return list(__segmenter_handler.keys())

def load_segmenter(
    seg_name : str,
    engine : Engine,
    device : str,
    model : BaseModule,
    loss_fn : phmLoss,
    optimizer,
    experiment : Experiment,
    **kwargs
):
    if not seg_name in list_segmenters():
        msg = f'{seg_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __segmenter_handler[seg_name](
        engine=engine,
        device=device,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        experiment=experiment,
        **kwargs
    )

class BaseSegmenter:
    def __init__(
        self,
        engine : Engine,
        device : str,
        model : BaseModule,
        loss_fn : phmLoss,
        optimizer,
        experiment : Experiment,
        metrics : List[phm_Metric] = None,
        **kwargs
    ):
        self.engine = engine
        self.device = torch.device(device)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.experiment = experiment
        self.metrics = metrics
        # Initialize the configuration
        for key, value in kwargs.items():
            setattr(self, key, value)

    def segment(self, batch):
        pass

    def __call__(self, batch) -> Any:
        transform = T.ToPILImage()
        # Log the image and target
        if self.engine.state.log_image:
            target = np.asarray(transform(torch.squeeze(batch[1])))
            img = np.asarray(transform(torch.squeeze(batch[0])))

            self.experiment.log_image(img, 
                overwrite=True,
                name=f'original', 
                step=self.engine.state.iteration)
            
            if target is not None:
                self.experiment.log_image(target, 
                    overwrite=True,
                    name=f'target', 
                    step=self.engine.state.iteration)

        # Recall the step
        res = self.segment(batch)

        out = np.asarray(transform(res.output)) if isinstance(res.output, torch.Tensor) else res.output
        out_ready = np.asarray(transform(res.output_ready)) if isinstance(res.output_ready, torch.Tensor) else res.output_ready

        if self.engine.state.log_metrics:
            if res.internal_metrics is not None and res.internal_metrics:
                self.experiment.log_metrics(res.internal_metrics, prefix='loop_',
                    step=self.engine.state.iteration, 
                    epoch=self.engine.state.epoch)
            if self.metrics is not None and self.metrics:
                targ = np.asarray(transform(res.target)) if isinstance(res.target, torch.Tensor) else res.target
                for m in self.metrics:
                    m.update((out, targ))
                    m.compute(self.experiment, prefix='step_',
                        step=self.engine.state.iteration, epoch=self.engine.state.epoch)

        if self.engine.state.log_image:
            self.experiment.log_image(out, 
                name=f'adapted_result', 
                step=self.engine.state.iteration)
            self.experiment.log_image(out_ready, 
                name=f'result', 
                step=self.engine.state.iteration)

