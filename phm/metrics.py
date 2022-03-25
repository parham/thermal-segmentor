

import numpy as np
from abc import ABC, abstractmethod

quality_metrics = {}
def quality_metric(method):
    global quality_metrics

    def _qembed(name, method):
        quality_metrics[name] = method

def _assert_image_shapes_equal(org: np.ndarray, pred: np.ndarray, metric: str):
    """ Shape of the image should be like this (rows, cols, bands)
        Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
        image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
        in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot
    
    Based on: https://github.com/up42/image-similarity-measures
    Args:
        org_img (np.ndarray): original image
        pred_img (np.ndarray): predicted image
        metric (str): _description_
    """

    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org.shape)}, y_pred shape = {str(pred.shape)}"
    )

    assert org.shape == pred.shape, msg

@quality_metric('RMSE')
def rmse(org: np.ndarray, pred: np.ndarray, max_p: int = 4095) -> float:
    """rmse : Root Mean Squared Error Calculated individually for all bands, then averaged
    Based on: https://github.com/up42/image-similarity-measures

    Args:
        org (np.ndarray): original image
        pred (np.ndarray): predicted image
        max_p (int, optional): maximum possible value. Defaults to 4095.

    Returns:
        float: RMSE value
    """
    _assert_image_shapes_equal(org, pred, "RMSE")

    rmse_bands = []
    for i in range(org_img.shape[2]):
        dif = np.subtract(org_img[:, :, i], pred_img[:, :, i])
        m = np.mean(np.square(dif / max_p))
        s = np.sqrt(m)
        rmse_bands.append(s)

    return np.mean(rmse_bands)

class Metric (object):

    def __init__(self) -> None:
        self.__name = 'metric'
    
    def __call__(self, src, dsc):
        return self._process(src, dsc)

    @abstractmethod
    def _process(src, dsc):
        pass

    def __str__(self) -> str:
        return self._name
     