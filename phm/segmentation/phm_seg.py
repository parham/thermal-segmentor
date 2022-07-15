

from typing import Dict, List

from comet_ml import Experiment

from ignite.engine import Engine
from ignite.engine.events import Events

from phm.metrics import phm_Metric
from phm.segmentation.core import segmenter_method


@segmenter_method(['phm_semisupr2022'])
def phm_semisupervised_segment(
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
    