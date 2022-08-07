



import functools
import time
from typing import Dict, List

from comet_ml import Experiment
from phm.loss import UnsupervisedLoss_TwoFactors, WNetLoss
from phm.metrics import phmMetric
from phm.models.wnet import WNet
from phm.postprocessing import adapt_output
from phm.segmentation.core import SegmentRecord, segmenter_method, simplify_train_step

import torch
import torch.nn as nn
import torch.optim as optim

from ignite.engine import Engine
from ignite.engine.events import Events

@segmenter_method(['phm_wnet'])
def wnet_segment(
    handler : str,
    data_name : str,
    category : Dict,
    experiment : Experiment,
    config = None,
    device : str = None,
    metrics : List[phmMetric] = None
) -> Engine:

    model = WNet(
        num_channels=config.model.num_channels,
        num_classes=len(category.keys())
    )
    model.to(device)
    loss = WNetLoss()

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

    # Event Handlers
    @engine.on(Events.ITERATION_COMPLETED)
    def __train_iteration_ended(engine):
        if engine.state.class_count <= engine.state.min_classes:
            engine.terminate()
    
    # return engine
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

    # img = img_data.squeeze(dim=0)
    img = img_data
    # target = target_data.squeeze(dim=0)
    target = target_data

    img_w = img.shape[0]
    img_h = img.shape[1]

    t = time.time()
    loss_fn.prepare_loss(ref=img)

    model.train()
    optimizer.zero_grad()

    output, mask = model(img)

    # Determine the number of classes in output
    nLabels = len(torch.unique(target))
    engine.state.class_count = nLabels

    loss = loss_fn(output=output, target=target, img_size=img.shape)
    loss.backward()
    engine.state.last_loss = loss.item()

    engine.state.step_time = time.time() - t

    optimizer.step()

    output_res = adapt_output(output, target, iou_thresh=engine.state.iou_thresh)

    return SegmentRecord(
        output=output_res[0], 
        target=target, 
        output_ready=output,
        loss=loss)