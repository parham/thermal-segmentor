
""" 
    @name classics.py   
    @info   classics.py provides classical segmentation algorithm
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import functools
from typing import Dict, List
from dotmap import DotMap
import numpy as np

from comet_ml import Experiment
from torchmetrics import Metric

from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from skimage import segmentation, color
from skimage.future import graph

from ignite.engine import Engine

from phm.core import load_config
from phm.metrics import phm_Metric
from phm.segment import Segmentor, ignite_segmenter

class ClassicSegmentor(Segmentor):
    def __init__(self, 
        experiment: Experiment = None,
        metrics : List[phm_Metric] = None
    ) -> None:
        super().__init__(experiment, metrics=metrics)

    def segment(self, img, target = None,
        log_img: bool = True,
        log_metrics: bool = True):
        return

    def segment_ignite__(self, engine, batch,
        log_img: bool = True,
        log_metrics: bool = True):
        img = batch[0]
        target = batch[1] if len(batch) > 1 else None
        return self.segment(img, target, log_img = log_img, log_metrics = log_metrics)

class DBSCAN_Impl(ClassicSegmentor):
    """ Implementation of DBSCAN inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """

    def __init__(self,
        eps: float = 0.5,
        min_samples: int = 5,
        leaf_size: int = 30,
        experiment: Experiment = None,
        metrics : List[phm_Metric] = None
    ) -> None:
        super().__init__(experiment, metrics=metrics)
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size

    def segment(self, img, target = None,
        log_img: bool = True,
        log_metrics: bool = True):

        if log_img:
            self.experiment.log_image(
                img, name='original', step=0)
        data = np.array(img)
        data = np.float32(data.reshape((-1, 3)))
        db = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            leaf_size=self.leaf_size
        ).fit(data[:, :2])
        seg_result = np.uint8(db.labels_.reshape(img.shape[:2]))
        
        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=0)
        stats = {}
        if log_metrics and \
            self.metrics is not None and \
            target is not None:
                for m in self.metrics:
                    m.update((seg_result, target))
                    stats['step_' + m.get_name()] = m.compute()
                    m.reset()
                self.experiment.log_metrics({**stats}, step=1, epoch=1)

        return 0.000001, seg_result

class KMeans_Impl(ClassicSegmentor):
    """ Implementation of KMeans inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """

    def __init__(self,
        dominant_colors: int = 4,
        experiment: Experiment = None,
        metrics : List[phm_Metric] = None
    ) -> None:
        super().__init__(experiment, metrics=metrics)
        self.n_clusters = dominant_colors

    def segment(self, img, target = None,
        log_img: bool = True,
        log_metrics: bool = True):

        if log_img:
            self.experiment.log_image(
                img, name='original', step=0)

        data = np.array(img)
        data = np.float32(data.reshape((-1, 3)))
        db = KMeans(n_clusters=self.n_clusters).fit(data[:, :2])
        seg_result = np.uint8(db.labels_.reshape(img.shape[:2]))
        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=0)
        stats = {}
        if log_metrics and \
            self.metrics is not None and \
            target is not None:
                for m in self.metrics:
                    m.update((seg_result, target))
                    stats['step_' + m.get_name()] = m.compute()
                    m.reset()
                self.experiment.log_metrics({**stats}, step=1, epoch=1)
        return 0.000001, seg_result

class MeanShift_Impl(ClassicSegmentor):
    """ Implementation of MeanShift inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """

    def __init__(self,
        quantile: float = 0.2,
        n_samples: int = 500,
        experiment: Experiment = None,
        metrics : List[phm_Metric] = None
    ) -> None:
        super().__init__(experiment, metrics=metrics)
        self.quantile = quantile
        self.n_samples = n_samples

    def segment(self, img, target = None,
        log_img: bool = True,
        log_metrics: bool = True):

        if log_img:
            self.experiment.log_image(
                img, name='original', step=0)

        data = np.array(img)
        data = np.float32(data.reshape((-1, 3)))
        bandwidth = estimate_bandwidth(
            data, quantile=self.quantile, n_samples=self.n_samples)
        db = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data[:, :2])
        seg_result = np.uint8(db.labels_.reshape(img.shape[:2]))
        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=0)
        stats = {}
        if log_metrics and \
            self.metrics is not None and \
            target is not None:
                for m in self.metrics:
                    m.update((seg_result, target))
                    stats['step_' + m.get_name()] = m.compute()
                    m.reset()
                self.experiment.log_metrics({**stats}, step=1, epoch=1)

        return 0.000001, seg_result

class GraphCut_Impl(ClassicSegmentor):
    """ Implementation of GraphCut inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """

    def __init__(self,
        compactness: int = 30,
        n_segments: int = 20000,
        experiment: Experiment = None,
        metrics : List[phm_Metric] = None
    ) -> None:
        super().__init__(experiment, metrics=metrics)
        self.compactness = compactness
        self.n_segments = n_segments

    def segment(self, img, target = None,
        log_img: bool = True,
        log_metrics: bool = True):

        if log_img:
            self.experiment.log_image(
                img, name='original', step=0)

        data = np.array(img)
        labels = segmentation.slic(
            data, compactness=self.compactness, n_segments=self.n_segments)
        # out_slic = color.label2rgb(labels, data, kind='avg')
        g = graph.rag_mean_color(data, labels, mode='similarity')
        labels = graph.cut_normalized(labels, g)
        seg_result = color.label2rgb(labels, data, kind='avg')

        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=0)
        stats = {}
        if log_metrics and \
            self.metrics is not None and \
            target is not None:
                for m in self.metrics:
                    m.update((seg_result, target))
                    stats['step_' + m.get_name()] = m.compute()
                    m.reset()
                self.experiment.log_metrics({**stats}, step=1, epoch=1)

        return 0.000001, seg_result

@ignite_segmenter(['dbscan', 'kmeans', 'meanshift', 'graphcut'])
def generate_classics_ignite__(
    name : str,
    config : DotMap,
    experiment : Experiment,
    metrics : List[phm_Metric] = None,
    step_metrics : List[phm_Metric] = None,
    **kwargs):

    seg_obj = None
    if name == 'dbscan':
        seg_obj = DBSCAN_Impl(
            eps=config.segmentation.eps,
            min_samples=config.segmentation.min_samples,
            leaf_size=config.segmentation.leaf_size,
            experiment=experiment,
            metrics=metrics)
    elif name == 'kmeans':
        seg_obj = KMeans_Impl(
            dominant_colors=config.segmentation.n_clusters,
            experiment=experiment,
            metrics=metrics)
    elif name == 'meanshift':
        seg_obj = MeanShift_Impl(
            quantile=config.segmentation.quantile,
            n_samples=config.segmentation.n_samples,
            experiment=experiment,
            metrics=metrics)
    elif name == 'graphcut':
        seg_obj = GraphCut_Impl(
            compactness = config.segmentation.compactness,
            n_segments = config.segmentation.n_segments,
            experiment=experiment,
            metrics=metrics)

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics 
    )

    return seg_obj, pred_func