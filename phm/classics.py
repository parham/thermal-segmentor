
import numpy as np

from comet_ml import Experiment
from phm import Segmentor

from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from skimage import data, segmentation, color
from skimage.future import graph

class DBSCAN_Impl(Segmentor):
    """ Implementation of DBSCAN inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """
    def __init__(self, 
        eps : float = 0.5,
        min_samples : int = 5,
        leaf_size : int = 30,
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
        data = np.float32(data.reshape((-1,3)))
        db = DBSCAN(
            eps=self.eps, 
            min_samples=self.min_samples,
            leaf_size=self.leaf_size
        ).fit(data[:,:2])
        seg_result = np.uint8(db.labels_.reshape(img.shape[:2]))
        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=0)
        return 0.000001, seg_result


class KMeans_Impl(Segmentor):
    """ Implementation of KMeans inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """
    def __init__(self, 
        dominant_colors : int = 4,
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
        data = np.float32(data.reshape((-1,3)))
        db = KMeans(n_clusters = self.n_clusters).fit(data[:,:2])
        seg_result = np.uint8(db.labels_.reshape(img.shape[:2]))
        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=0)
        return 0.000001, seg_result


class MeanShift_Impl(Segmentor):
    """ Implementation of MeanShift inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """
    def __init__(self, 
        quantile : float = 0.2,
        n_samples : int = 500,
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
        data = np.float32(data.reshape((-1,3)))
        bandwidth = estimate_bandwidth(data, quantile=self.quantile, n_samples=self.n_samples)
        db = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data[:,:2])
        seg_result = np.uint8(db.labels_.reshape(img.shape[:2]))
        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=0)
        return 0.000001, seg_result

class GraphCut_Impl(Segmentor):
    """ Implementation of GraphCut inspired by https://github.com/charmichokshi/Unsupervised-Image-Segmentation-Algorithms
    """
    def __init__(self, 
        compactness : int = 30,
        n_segments : int = 20000,
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
        labels = segmentation.slic(data, compactness=self.compactness, n_segments=self.n_segments)
        # out_slic = color.label2rgb(labels, data, kind='avg')
        g = graph.rag_mean_color(data, labels, mode='similarity')
        labels = graph.cut_normalized(labels, g)
        seg_result = color.label2rgb(labels, data, kind='avg')

        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=0)
        return 0.000001, seg_result
