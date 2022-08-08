
import torch
import torchvision.transforms as T

import logging
import numpy as np

from comet_ml import Experiment
from ignite.engine import Engine
from typing import Any, Callable, Dict, List, NamedTuple, Union

from phm.core import phmCore
from phm.metrics import BaseMetric

label_colors_1ch8bits = np.random.randint(10,255,size=(100,1))

__segmenter_handler = {}

class SegmentRecord(NamedTuple):
    iteration : int
    orig : Any
    loss : float
    output : Any
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

class BaseSegmenter(phmCore):
    def __init__(
        self,
        name : str,
        device : str,
        config : Dict[str,Any],
        experiment : Experiment,
        metrics : List[BaseMetric],
        preprocess : Callable = None,
        postprocess : Callable = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            device=device,
            config=config
        )

        self.experiment = experiment
        self.metrics = metrics
        # Preprocess step
        self.preprocess = preprocess
        # Postprocess step
        self.postprocess = postprocess

        # Add fields dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize Engine
        self.engine = Engine(self.__call__)
        self.__init_state(config)

    def __init_state(self, config):
        # Add configurations to the engine state
        for sec in config.keys():
            for key, value in config[sec].items():
                self.engine.state_dict_user_keys.append(key)
                setattr(self.engine.state, key, value)
        # Status
        self.engine.state_dict_user_keys.append('class_count')
        self.engine.state.class_count = 0
        self.engine.state_dict_user_keys.append('last_loss')
        self.engine.state.last_loss = 0

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

def load_segmenter(
    seg_name : str,
    device : str,
    config : Dict[str, Any],
    experiment : Experiment,
    metrics : List[BaseMetric],
    preprocess : Callable = None,
    postprocess : Callable = None,
    **kwargs
) -> BaseSegmenter:
    if not seg_name in list_segmenters():
        msg = f'{seg_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __segmenter_handler[seg_name](
        name=seg_name,
        device=device,
        config=config,
        experiment=experiment,
        metrics=metrics,
        preprocess=preprocess,
        postprocess=postprocess,
        **kwargs
    )

