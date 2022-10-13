
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
import copy
from comet_ml import Experiment
from typing import Callable, Dict
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from ignite.engine import Engine

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.core import get_device, get_profile, make_tensor_for_comet
from lemanchot.pipeline.core import pipeline_register
from lemanchot.processing import adapt_output, classmap_2_multilayer_numpy
from lemanchot.pipeline.wrapper import BaseWrapper, wrapper_register

@wrapper_register('iterative_wrapper')
class IterativeWrapper(BaseWrapper):
    def __init__(self, 
        step_func: Callable, 
        device: str, 
        **kwargs
    ) -> None:
        super().__init__(step_func, device, **kwargs)
        self.ref_model = copy.deepcopy(self.model)
    
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
        labels = batch[-1]
        # Logging computation time
        t = time.time()
        # Apply the model to data
        res = None
        cat_num = len(profile.categories)
        
        pbar = tqdm(range(engine.state.max_iteration))
        model.load_state_dict(self.ref_model.state_dict())
        for i in pbar:
            res = step_func(
                engine=engine,
                batch=data,
                device=device,
                model=model,
                criterion=loss,
                optimizer=optimizer,
                experiment=experiment,
            )
            pbar.set_description('Number of Classes %d' % res['num_classes'])
            if res['num_classes'] <= cat_num:
                break

        step_time = time.time() - t

        # Logging loss & step time
        if "loss" in res:
            engine.state.metrics["loss"] = res["loss"]
        if 'num_classes' in res:
            engine.state.metrics["num_classes"] = res["num_classes"]
        engine.state.metrics["step_time"] = step_time

        targets = res["y_true"]
        outputs = res["y_pred"]
        
        trg = targets.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.uint8)
        out = outputs.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.uint8)
        out, _, _ = adapt_output(out, trg, engine.state.iou_thresh)
        # Convert to multilayer
        out = classmap_2_multilayer_numpy(out, classes=list(profile.categories.values()))
        trg = classmap_2_multilayer_numpy(trg, classes=list(profile.categories.values()))
        
        out = torch.Tensor(out).to(device=get_device()).to(dtype=torch.int)
        trg = torch.Tensor(trg).to(device=get_device()).to(dtype=torch.int)
        
        out = torch.permute(out.unsqueeze(0), [0,-1,1,2])
        trg = torch.permute(trg.unsqueeze(0), [0,-1,1,2])
        res['y_res'] = out

        # Calculate metrics
        for m in metrics:
            m.update((out, trg))
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
                        lbl = labels[i]
                        sample = make_tensor_for_comet(img[i, :, :, :], coloring = False)
                        label = f"{lbl}-{i}"
                        experiment.log_image(sample, f'{lbl}-{key_lbl}', step=engine.state.iteration)
            
            # Save all samples in a batch
            if img_saver is not None:
                record = {'labels' : labels}
                record['metrics'] = engine.state.metrics
                for key, img in res.items():
                    tmp = img.cpu().detach().numpy() if isinstance(img, torch.Tensor) else img
                    record[key] = tmp
                img_saver(f'{engine.state.epoch}-{engine.state.iteration}', record)
        
        return res