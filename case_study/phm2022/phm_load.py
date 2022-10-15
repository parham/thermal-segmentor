
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Union
from uuid import uuid4
from dotmap import DotMap

import numpy as np
import torch
import torch.optim as optim

from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.pipeline.core import load_optimizer, load_pipeline, load_scheduler
from lemanchot.core import (
    exception_logger,
    get_config,
    get_device,
    get_experiment,
    get_or_default,
    get_profile,
    load_settings,
)
from lemanchot.loss import load_loss
from lemanchot.loss.core import load_loss_inline__
from lemanchot.metrics import load_metrics
from lemanchot.models import BaseModule, load_model
from lemanchot.pipeline.saver import ImageSaver, MatSaver, ModelLogger_CometML
from lemanchot.pipeline.wrapper import load_wrapper

@exception_logger
def load_phm2022_segmentation(profile_name: str, database_name: str) -> Dict:
    # Load experiment configuration
    experiment_name = get_profile(profile_name).experiment_config_name
    experiment_config = get_config(experiment_name)
    device = get_device()
    ############ Deep Model ##############
    # Create model instance
    model = load_model(experiment_config)
    if model is not None:
        model.to(device)
    ############ Loss function ##############
    # Create loss instance
    # Get the loss configuration
    loss_config = experiment_config.loss
    loss = {}
    for name, cfg in loss_config.items():
        loss[name] = load_loss_inline__(cfg.name, cfg.config)
    ############ Optimizer ##############
    # Create optimizer instance
    opt_cfg = experiment_config.optimizer['phm_nolabel']
    optim_nolbl = load_optimizer(model.get_model('phm_unsupervised'), DotMap({'optimizer' : opt_cfg}))
    
    opt_cfg = experiment_config.optimizer['phm_label']
    optim_lbl = load_optimizer(model.get_model('phm_unsupervised'), DotMap({'optimizer' : opt_cfg}))
    
    opt_cfg = experiment_config.optimizer['phm_supervised']
    optim_sup = load_optimizer(model.get_model('phm_supervised'), DotMap({'optimizer' : opt_cfg}))
        
    optimizer = {
        'phm_nolabel' : optim_nolbl,
        'phm_label' : optim_lbl,
        'phm_supervised' : optim_sup
    }
    ############ Comet.ml Experiment ##############
    # Create the experiment instance
    experiment = get_experiment(profile_name=profile_name, dataset=database_name)
    # Logging the model
    experiment.set_model_graph(str(model), overwrite=True)
    # Load profile
    profile = get_profile(profile_name)
    # Load the pipeline
    pipeline_name = profile.pipeline
    if not pipeline_name in experiment_config.pipeline:
        raise ValueError("Pipeline is not supported!")
    ############ Pipeline ##############
    # Get Pipeline Configuration
    pipeline_config = experiment_config.pipeline[pipeline_name]
    # Load Pipeline Handler
    step_func = load_pipeline(pipeline_name)
    ############ Metrics ##############
    # Create metric instances
    metrics = load_metrics(experiment_config, profile.categories)
    # Create the image logger
    img_saver = None
    if "image_saving" in profile:
        image_saving = profile.image_saving
        img_saver = MatSaver(**image_saving)

    wrapper_name = pipeline_config.wrapper
    seg_func = load_wrapper(
        wrapper_name=wrapper_name,
        step_func=step_func,
        device=device,
        model=model,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        experiment=experiment,
        img_saver=img_saver,
    )

    # Log hyperparameters
    experiment.log_parameters(experiment_config.toDict())
    # Instantiate the engine
    engine = Engine(seg_func)
    # Create scheduler instance for Supervised Module
    scheduler = load_scheduler(engine, optim_sup, experiment_config)
    # Add configurations to the engine state
    engine.state.last_loss = 0
    if experiment_config.pipeline:
        for key, value in pipeline_config.items():
            engine.state_dict_user_keys.append(key)
            setattr(engine.state, key, value)
    engine.state.profile_name = profile_name

    run_record = {
        "engine": engine,
        "model": model,
        "optimizer": optimizer,
        "loss": loss,
    }
    enable_checkpoint_save = get_or_default(profile, "checkpoint_save", False)
    enable_checkpoint_log = get_or_default(profile, "checkpoint_log_cometml", False)

    checkpoint_file = f"{pipeline_name}-{model.name}-{str(uuid4())[0:8]}.pt"
    if enable_checkpoint_save:
        experiment.log_parameter(name="checkpoint_file", value=checkpoint_file)
        checkpoint_dir = load_settings().checkpoint_dir
        checkpoint_file = checkpoint_file
        checkpoint_saver = ModelCheckpoint(
            dirname=checkpoint_dir,
            filename_pattern=checkpoint_file,
            filename_prefix="",
            require_empty=False,
            create_dir=True,
            n_saved=1,
            global_step_transform=global_step_from_engine(engine),
        )
        engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver, run_record)
        # Logging Model
        if enable_checkpoint_log:
            checkpoint_logger = ModelLogger_CometML(
                pipeline_name, model.name, experiment, checkpoint_saver
            )
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, checkpoint_logger, run_record
            )

    # Load Checkpoint
    enable_checkpoint_load = get_or_default(profile, "checkpoint_load", False)
    if enable_checkpoint_load:
        checkpoint_dir = load_settings().checkpoint_dir
        checkpoint_file = os.path.join(checkpoint_dir, f"{profile.checkpoint_file}")
        if os.path.isfile(checkpoint_file):
            checkpoint_obj = torch.load(checkpoint_file, map_location=get_device())
            if profile.load_weights_only:
                run_record["model"].load_state_dict(checkpoint_file["model"])
            else:
                ModelCheckpoint.load_objects(
                    to_load=run_record, checkpoint=checkpoint_obj
                )

    @engine.on(Events.ITERATION_COMPLETED(every=1))
    def __log_training(engine):
        lr = engine.state.metrics["lr"] if "lr" in engine.state.metrics else 0
        epoch = engine.state.epoch
        max_epochs = engine.state.max_epochs
        iteration = engine.state.iteration
        step_time = engine.state.step_time if hasattr(engine.state, "step_time") else 0
        vloss = get_or_default(engine.state.metrics, "loss", 0)
        print(
            f"Epoch {epoch}/{max_epochs} [{step_time}] : {iteration} - batch loss: {vloss:.4f}, lr: {lr:.4f}"
        )

    @engine.on(Events.ITERATION_COMPLETED(every=1))
    def __log_metrics(engine):
        profile = get_profile(engine.state.profile_name)
        if profile.enable_logging:
            metrics = engine.state.metrics
            experiment.log_metrics(
                dict(metrics), step=engine.state.iteration, epoch=engine.state.epoch
            )

    @engine.on(Events.STARTED)
    def __train_process_started(engine):
        experiment.train()
        logging.info("Training started ...")

    @engine.on(Events.COMPLETED)
    def __train_process_ended(engine):
        logging.info("Training ended ...")
        experiment.end()

    return run_record
