
import numpy as np

from comet_ml import Experiment
from phm import Segmentor

from sklearn.cluster import DBSCAN

class DBSCAN_Impl(Segmentor):
    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)

    def segment_noref(self, img,
        log_img: bool = True,
        log_metrics: bool = True):

        if log_img:
            self.experiment.log_image(
                img, name='original', step=0)
        data = np.array(img)
        data = np.float32(data.reshape((-1,3)))
        db = DBSCAN(eps=1.2, min_samples=70).fit(data[:,:2])
        seg_result = np.uint8(db.labels_.reshape(img.shape[:2]))
        if log_img:
            self.experiment.log_image(
                seg_result, name='steps', step=0)
        return 0.000001, seg_result