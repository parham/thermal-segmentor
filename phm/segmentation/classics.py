
import time
import numpy as np

from typing import Any, Dict, List
from comet_ml import Experiment
import torch
from ignite.engine import Engine

from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from skimage import segmentation, color
from skimage.future import graph
from phm.loss.core import BaseLoss

from phm.metrics import BaseMetric
from phm.models.core import BaseModule
from phm.segmentation.core import BaseSegmenter, SegmentRecord, segmenter_method, label_colors_1ch8bits, simplify_train_step
from phm.postprocessing import remove_small_regions, adapt_output

@segmenter_method('dbscan')
class DBScanSegmenter(BaseSegmenter):
    def __init__(
        self,
        name : str,
        device : str,
        config : Dict[str,Any],
        experiment : Experiment,
        metrics : List[BaseMetric],
        **kwargs
    ):
        super().__init__(
            name=name,
            device=device,
            config=config,
            experiment=experiment,
            metrics=metrics,
            **kwargs
        )
    
    def segment(self, batch):
        result = None
        img_data = batch[0]
        target_data = batch[1] if len(batch) > 1 else None

        t = time.time()

        img = img_data.squeeze(dim=0)
        target = target_data.squeeze(dim=0)

        data = np.array(img)
        data = np.float32(data.reshape((-1, 3)))
        db = DBSCAN(
            eps=self.engine.state.eps,
            min_samples=self.engine.state.min_samples,
            leaf_size=self.engine.state.leaf_size
        ).fit(data[:, :2])
        result = np.uint8(db.labels_.reshape(img.shape[:2]))

        self.engine.state.step_time = time.time() - t

        nLabels = len(np.unique(result))
        self.engine.state.class_count = nLabels
        self.engine.state.last_loss = 0.00001


@segmenter_method(['dbscan', 'kmeans', 'meanshift', 'graphcut'])
def classical_segment(
    data_name : str,
    handler : str,
    category : Dict,
    experiment : Experiment,
    config = None,
    device : str = None,
    metrics : List[BaseMetric] = None
) -> Engine:
    
    def __train_step(engine, batch):
        result = None
        img_data = batch[0]
        target_data = batch[1] if len(batch) > 1 else None

        img = img_data.squeeze(dim=0)
        target = target_data.squeeze(dim=0)

        def __dbscan(img, target):
            data = np.array(img)
            data = np.float32(data.reshape((-1, 3)))
            db = DBSCAN(
                eps=engine.state.eps,
                min_samples=engine.state.min_samples,
                leaf_size=engine.state.leaf_size
            ).fit(data[:, :2])
            result = np.uint8(db.labels_.reshape(img.shape[:2]))
            return result
        
        def __kmeans(img, target):
            data = np.array(img)
            data = np.float32(data.reshape((-1, 3)))
            db = KMeans(n_clusters=engine.state.n_clusters).fit(data[:, :2])
            result = np.uint8(db.labels_.reshape(img.shape[:2]))
            return result
        
        def __mean_shift(img, target):
            data = np.array(img)
            data = np.float32(data.reshape((-1, 3)))
            bandwidth = estimate_bandwidth(data, 
                quantile=engine.state.quantile, 
                n_samples=engine.state.n_samples)
            db = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data[:, :2])
            result = np.uint8(db.labels_.reshape(img.shape[:2]))
            return result

        def __graph_cut(img, target):
            data = np.array(img)
            labels = segmentation.slic(data, 
                compactness=engine.state.compactness, 
                n_segments=engine.state.n_segments)
            # out_slic = color.label2rgb(labels, data, kind='avg')
            g = graph.rag_mean_color(data, labels, mode='similarity')
            labels = graph.cut_normalized(labels, g)
            result = color.label2rgb(labels, data, kind='avg')
            return result
        
        t = time.time()
        result = {
            'dbscan' : __dbscan,
            'kmeans' : __kmeans,
            'meanshift' : __mean_shift,
            'graphcut' : __graph_cut
        }[handler](img, target)

        engine.state.step_time = time.time() - t

        result = __helper_postprocessing(engine, result)
        nLabels = len(np.unique(result))
        engine.state.class_count = nLabels
        engine.state.last_loss = 0.00001

        target_np = target.cpu().detach().numpy() if target is not None else None
        return __helper_prepare_result(engine, img, result, target_np, internal_metrics={
            'step_time' : engine.state.step_time,
            'class_count' : engine.state.class_count
        })

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
    
    train_step = simplify_train_step(experiment, __train_step, metrics=metrics)

    engine = Engine(train_step)
    __init_state(config)

    # return engine
    return {
        'engine' : engine
    }

def __helper_postprocessing(engine, img):
        # Coloring regions
    im_color = np.array([label_colors_1ch8bits[ c % 255 ] for c in img]).reshape(img.shape).astype(np.uint8)
    # Small regions
    return remove_small_regions(im_color, min_area=engine.state.min_area)

def __helper_prepare_result(engine, input, output, target, internal_metrics : Dict = {}):
    output_res = adapt_output(output, target, iou_thresh=engine.state.iou_thresh)
    return SegmentRecord(
        output=output_res[0], 
        target=target, 
        output_ready=output,
        loss=0.00001,
        internal_metrics=internal_metrics)