
import torch
import torchvision.transforms as T

import logging
import numpy as np

from typing import Any, Callable, Dict, List, NamedTuple, Union
from comet_ml import Experiment
from ignite.engine import Engine

from phm.loss import phmLoss
from phm.metrics import phmMetric
from phm.models import BaseModule
from phm.postprocessing import adapt_output, remove_small_regions

label_colors_1ch8bits = np.random.randint(10,255,size=(100,1))

__segmenter_handler = {}

class SegmentRecord(NamedTuple):
    orig : Any
    loss : float
    orig_output : Any
    processed_output : Any
    target : Any = None
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
    device : str,
    model : BaseModule,
    loss_fn : phmLoss,
    optimizer,
    experiment : Experiment,
    config
):
    if not seg_name in list_segmenters():
        msg = f'{seg_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __segmenter_handler[seg_name](
        device=device,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        experiment=experiment,
        config=config
    )

class BaseSegmenter:
    def __init__(
        self,
        model : BaseModule,
        loss_fn : phmLoss,
        optimizer,
        config : Dict,
        experiment : Experiment,
        metrics : List[phmMetric] = None,
        device : str = 'gpu'
    ):
        self.device = torch.device(device)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.experiment = experiment
        self.metrics = metrics
        # Initialize the configuration
        for key, value in config.items():
            setattr(self, key, value)
        self.engine = Engine(self.__call__)

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

        orig_output = np.asarray(transform(res.orig_output)) if isinstance(res.orig_output, torch.Tensor) else res.orig_output
        processed_output = np.asarray(transform(res.processed_output)) if isinstance(res.processed_output, torch.Tensor) else res.processed_output

        if self.engine.state.log_metrics:
            if res.internal_metrics is not None and res.internal_metrics:
                self.experiment.log_metrics(res.internal_metrics, prefix='loop_',
                    step=self.engine.state.iteration, 
                    epoch=self.engine.state.epoch)
            if self.metrics is not None and self.metrics:
                targ = np.asarray(transform(res.target)) if isinstance(res.target, torch.Tensor) else res.target
                for m in self.metrics:
                    m.update((processed_output, targ))
                    m.compute(self.experiment, prefix='step_',
                        step=self.engine.state.iteration, epoch=self.engine.state.epoch)

        if self.engine.state.log_image:
            self.experiment.log_image(processed_output, 
                name=f'processed_result', 
                step=self.engine.state.iteration)
            self.experiment.log_image(orig_output, 
                name=f'orig_result', 
                step=self.engine.state.iteration)

