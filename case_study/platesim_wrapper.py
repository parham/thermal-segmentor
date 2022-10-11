
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import os
import sys
import time
from comet_ml import Experiment
from typing import Callable, Dict

import torch
import torch.optim as optim
from torchvision.transforms import ToPILImage
from ignite.engine import Engine

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.core import get_device, get_profile, make_tensor_for_comet
from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule
from lemanchot.pipeline.wrapper import BaseWrapper, wrapper_register

@wrapper_register('platesim_wrapper')
class PlateSimWrapper(BaseWrapper):
    def __init__(self, 
        step_func: Callable, 
        device: str, 
        **kwargs
    ) -> None:
        super().__init__(step_func, device, **kwargs)
        self.to_pil = ToPILImage()
    
    def __call__(self,
        engine: Engine,
        batch
    ) -> Dict:

        # Local Variable Initialization
        step_func = self.step_func
        device = self.device
        model = self.model
        loss = self.loss
        optimizer = self.optimizer
        metrics = self.metrics
        experiment = self.experiment
        img_saver = self.img_saver if hasattr(self, 'img_saver') else None

        profile = get_profile(engine.state.profile_name)

        data = list(map(lambda x: x.to(device=get_device()), batch[0:2]))
        label_str = batch[-1][0]
        # Logging computation time
        t = time.time()
        # Apply the model to data
        res = step_func(
            engine=engine,
            batch=data,
            device=device,
            model=model,
            criterion=loss,
            optimizer=optimizer,
            experiment=experiment,
        )
        step_time = time.time() - t

        # Logging loss & step time
        if "loss" in res:
            engine.state.metrics["loss"] = res["loss"]
        engine.state.metrics["step_time"] = step_time

        targets = res["y_true"]
        outputs = res["y_pred"] if not "y_processed" in res else res["y_processed"]

        # Calculate metrics
        for m in metrics:
            m.update((outputs, targets))
            m.compute(engine, experiment)

        if profile.enable_logging:
            # Calculate metrics
            if "metrics" in res:
                engine.state.metrics.update(res["metrics"])

            # Assume Tensor B x C x W x H
            # Logging imagery results
            for key, img in res.items():
                # if key == 'y_':
                if 'y_' in key:
                    key_lbl = key.replace('y_','')
                    # Control number of logged images with enable_image_logging setting.
                    for i in range(min(profile.enable_image_logging, img.shape[0])):
                        sample = make_tensor_for_comet(img[i, :, :, :], coloring = False)
                        label = f"{label_str}-{i}"
                        experiment.log_image(sample, f'{label_str}-{key_lbl}', step=engine.state.iteration)

            # Save all samples in a batch
            if img_saver is not None:
                record = {'labels' : label_str}
                record['metrics'] = engine.state.metrics
                for key, img in res.items():
                    tmp = img.cpu().detach().numpy() if isinstance(img, torch.Tensor) else img
                    record[key] = tmp
                img_saver(f'{engine.state.epoch}-{engine.state.iteration}', record)

        return res