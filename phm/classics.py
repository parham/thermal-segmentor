
import functools
from typing import Dict
from dotmap import DotMap
import numpy as np

from comet_ml import Experiment
from torchmetrics import Metric
from phm import Segmentor

from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from skimage import segmentation, color
from skimage.future import graph

from ignite.engine import Engine

from phm.core import load_config
from phm.segment import ignite_segmenter

class ClassicSegmentor(Segmentor):
    def __init__(self, experiment: Experiment = None) -> None:
        super().__init__(experiment)

    def segment_noref(self, img,
        log_img: bool = True,
        log_metrics: bool = True):
        pass

    def segment_noref_ignite__(self, engine, batch,
        log_img: bool = True,
        log_metrics: bool = True):
        img = batch[0]
        return self.segment_noref(img, log_img = log_img, log_metrics = log_metrics)


class DBSCAN_Impl(ClassicSegmentor):
    """ Implementation of DBSCAN inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """

    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 leaf_size: int = 30,
                 experiment: Experiment = None
                 ) -> None:
        super().__init__(experiment)
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size

    def segment_noref(self, img,
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
        return 0.000001, seg_result

class KMeans_Impl(ClassicSegmentor):
    """ Implementation of KMeans inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """

    def __init__(self,
                 dominant_colors: int = 4,
                 experiment: Experiment = None
                 ) -> None:
        super().__init__(experiment)
        self.n_clusters = dominant_colors

    def segment_noref(self, img,
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
        return 0.000001, seg_result

class MeanShift_Impl(ClassicSegmentor):
    """ Implementation of MeanShift inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """

    def __init__(self,
                 quantile: float = 0.2,
                 n_samples: int = 500,
                 experiment: Experiment = None
                 ) -> None:
        super().__init__(experiment)
        self.quantile = quantile
        self.n_samples = n_samples

    def segment_noref(self, img,
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
        return 0.000001, seg_result

class GraphCut_Impl(ClassicSegmentor):
    """ Implementation of GraphCut inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """

    def __init__(self,
                 compactness: int = 30,
                 n_segments: int = 20000,
                 experiment: Experiment = None
                 ) -> None:
        super().__init__(experiment)
        self.compactness = compactness
        self.n_segments = n_segments

    def segment_noref(self, img,
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
        return 0.000001, seg_result

def create_noref_predict_classics__(
    method_name : str,
    config_file : str, 
    experiment : Experiment = None,
    metrics : Dict[str,Metric] = None):

    config = load_config(config_file)

    obj = None
    if method_name == 'dbscan':
        obj = DBSCAN_Impl(
            eps=config.segmentation.eps,
            min_samples=config.segmentation.min_samples,
            leaf_size=config.segmentation.leaf_size,
            experiment=experiment)
    elif method_name == 'kmeans':
        obj = KMeans_Impl(
            dominant_colors=config.segmentation.n_clusters,
            experiment=experiment)
    elif method_name == 'meanshift':
        obj = MeanShift_Impl(
            quantile=config.segmentation.quantile,
            n_samples=config.segmentation.n_samples,
            experiment=experiment)
    elif method_name == 'graphcut':
        obj = GraphCut_Impl(
            compactness = config.segmentation.compactness,
            n_segments = config.segmentation.n_segments,
            experiment=experiment)

    if experiment is not None:
        experiment.log_parameters(config.segmentation, prefix='segmentation')

    pred_func = functools.partial(
        obj.segment_noref_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics 
    )
    engine = Engine(pred_func)

    if metrics is not None:
        for x in metrics.keys():
            metrics[x].attach(engine, x)

    return engine

@ignite_segmenter(['dbscan', 'kmeans', 'meanshift', 'graphcut'])
def generate_classics_ignite__(
    name : str,
    config : DotMap,
    experiment : Experiment):

    seg_obj = None
    if name == 'dbscan':
        seg_obj = DBSCAN_Impl(
            eps=config.segmentation.eps,
            min_samples=config.segmentation.min_samples,
            leaf_size=config.segmentation.leaf_size,
            experiment=experiment)
    elif name == 'kmeans':
        seg_obj = KMeans_Impl(
            dominant_colors=config.segmentation.n_clusters,
            experiment=experiment)
    elif name == 'meanshift':
        seg_obj = MeanShift_Impl(
            quantile=config.segmentation.quantile,
            n_samples=config.segmentation.n_samples,
            experiment=experiment)
    elif name == 'graphcut':
        seg_obj = GraphCut_Impl(
            compactness = config.segmentation.compactness,
            n_segments = config.segmentation.n_segments,
            experiment=experiment)

    pred_func = functools.partial(
        seg_obj.segment_noref_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics 
    )

    return seg_obj, pred_func