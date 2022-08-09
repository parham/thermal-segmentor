
import time
import numpy as np

from typing import Any, Callable, Dict, List
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

class ClassicSegmenter(BaseSegmenter):
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
            config=config,
            experiment=experiment,
            metrics=metrics,
            preprocess=preprocess,
            postprocess=postprocess if postprocess is not None else self.__postprocessing,
            **kwargs
        )

    def __postprocessing(self, res):
        # Coloring regions
        tmp = np.array([label_colors_1ch8bits[ c % 255 ] for c in res.output]).reshape(res.output.shape).astype(np.uint8)
        # Small regions
        res.processed_output = remove_small_regions(
            tmp, 
            min_area=self.engine.state.min_area
        )
        return res

    def segment(self, batch):
        img_data = batch[0]
        target_data = batch[1] if len(batch) > 1 else None

        t = time.time()

        img = img_data.squeeze(dim=0)
        target = target_data.squeeze(dim=0)

        self.loss_fn.prepare_loss(ref=img)

        output = self.segment_impl(img)

        loss = self.loss_fn(
            output=output,
            target=target
        )
        loss.backward()

        vloss = loss.item()
        nLabels = len(np.unique(output))
        self.engine.state.class_count = nLabels
        self.engine.state.last_loss = loss.item()
        self.engine.state.step_time = time.time() - t

        return SegmentRecord(
            iteration=self.engine.state.iteration,
            orig=img,
            output=output, 
            target=target,
            loss=vloss,
            internal_metrics={
                'loss' : vloss,
                'step_time' : self.engine.state.step_time,
                'class_count' : self.engine.state.class_count
            })

@segmenter_method('dbscan')
class DBScanSegmenter(ClassicSegmenter):
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
            config=config,
            experiment=experiment,
            metrics=metrics,
            preprocess=preprocess,
            postprocess=postprocess,
            **kwargs
        )

    def segment_impl(self,img):
        data = np.array(img)
        data = np.float32(data.reshape((-1, 3)))
        db = DBSCAN(
            eps=self.engine.state.eps,
            min_samples=self.engine.state.min_samples,
            leaf_size=self.engine.state.leaf_size
        ).fit(data[:, :2])
        output = np.uint8(db.labels_.reshape(img.shape[:2]))
        return output

@segmenter_method('kmeans')
class KMeanSegmenter(ClassicSegmenter):
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
            config=config,
            experiment=experiment,
            metrics=metrics,
            preprocess=preprocess,
            postprocess=postprocess,
            **kwargs
        )

    def segment_impl(self,img):
        data = np.array(img)
        data = np.float32(data.reshape((-1, 3)))
        db = KMeans(
            n_clusters=self.engine.state.n_clusters
        ).fit(data[:, :2])
        output = np.uint8(db.labels_.reshape(img.shape[:2]))
        return output

@segmenter_method('meanshift')
class MeanShiftSegmenter(ClassicSegmenter):
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
            config=config,
            experiment=experiment,
            metrics=metrics,
            preprocess=preprocess,
            postprocess=postprocess,
            **kwargs
        )

    def segment_impl(self,img):
        data = np.array(img)
        data = np.float32(data.reshape((-1, 3)))
        bandwidth = estimate_bandwidth(data, 
            quantile=self.engine.state.quantile, 
            n_samples=self.engine.state.n_samples)
        db = MeanShift(
            bandwidth=bandwidth, 
            bin_seeding=True
        ).fit(data[:, :2])
        output = np.uint8(db.labels_.reshape(img.shape[:2]))
        return output

@segmenter_method('graphcut')
class GraphcutSegmenter(ClassicSegmenter):
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
            config=config,
            experiment=experiment,
            metrics=metrics,
            preprocess=preprocess,
            postprocess=postprocess,
            **kwargs
        )

    def segment_impl(self,img):
        data = np.array(img)
        labels = segmentation.slic(data, 
            compactness=self.engine.state.compactness, 
            n_segments=self.engine.state.n_segments)
        g = graph.rag_mean_color(data, labels, mode='similarity')
        labels = graph.cut_normalized(labels, g)
        output = color.label2rgb(labels, data, kind='avg')
        return output
