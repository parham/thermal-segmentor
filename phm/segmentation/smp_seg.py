
import time
import functools
import numpy as np
from typing import Dict, List
from comet_ml import Experiment

import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToPILImage

from ignite.engine import Engine
from ignite.engine.events import Events

from phm.loss import FocalLoss
from phm.metrics import phm_Metric
from phm.segmentation.core import SegmentRecord, segmenter_method, simplify_train_step

__smp_unet_resnet18__ = 'smp_unet_resnet18'

@segmenter_method([__smp_unet_resnet18__])
def smp_seg(
    handler : str,
    data_name : str,
    category : Dict,
    experiment : Experiment,
    config = None,
    device : str = None,
    metrics : List[phm_Metric] = None
) -> Engine:
    model = None
    loss = None
    if handler == __smp_unet_resnet18__:
        model = smp.Unet(
            encoder_name='resnet18',
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(category.keys())+1 
        )
        loss = smp.losses.FocalLoss(mode = 'multiclass')
        # loss = FocalLoss()
    else:
        raise ValueError(f'{handler} is not supported!')
    
    model.to(device)
    # Logging the model
    experiment.set_model_graph(str(model), overwrite=True)
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(),
        lr=config.segmentation.learning_rate,
        momentum=config.segmentation.momentum
    )

    seg_func = functools.partial(
        segment_ignite__,
        model=model,
        loss_fn=loss,
        device=device,
        optimizer=optimizer,
        experiment=experiment
    )

    train_step = simplify_train_step(experiment, seg_func, metrics=metrics)
    engine = Engine(train_step)
    
    def __init_state(config):
        # Add configurations to the engine state
        for sec in config.keys():
            for key, value in config[sec].items():
                engine.state_dict_user_keys.append(key)
                setattr(engine.state, key, value)
        # Status
        engine.state_dict_user_keys.append('class_count')
        engine.state.class_count = 0
        engine.state_dict_user_keys.append('last_loss')
        engine.state.last_loss = 0

    __init_state(config)
    
    return {
        'engine' : engine,
        'model' : model,
        'optimizer' : optimizer
    }

def segment_ignite__(
    engine, batch,
    model, loss_fn, optimizer,
    experiment : Experiment,
    device
):
    result = None
    img_data = batch[0]
    target_data = batch[1] if len(batch) > 1 else None

    img = img_data.to(device=device, dtype=torch.float32)
    target = target_data.to(device=device, dtype=torch.float32)

    # Initialize training time
    img_h = img.shape[-2]
    img_w = img.shape[-1]
    t = time.time()

    model.train()
    optimizer.zero_grad()

    output = model(img)
    nLabels = output.shape[1]
    engine.state.class_count = nLabels

    loss = loss_fn(output, target.squeeze(dim=1))
    loss.backward()

    optimizer.step()
    
    transform = ToPILImage()

    engine.state.last_loss = loss.item()
    engine.state.step_time = time.time() - t

    output = np.asarray(transform(output.squeeze()))
    output = np.argmax(output, axis=2).astype(np.uint8)
    target = np.asarray(transform(target.squeeze()))

    return SegmentRecord(
        output=output,
        target=target,
        output_ready=output,
        loss=loss,
        internal_metrics={
            'loss' : loss,
            'step_time' : engine.state.step_time,
            'class_count' : engine.state.class_count
        })