

import cv2
import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod

def extract_regions(data : np.ndarray) -> List[Dict]:
    """Extract independent regions from segmented image

    Args:
        data (np.ndarray): segmented image which each pixel presented the class id.

    Returns:
        List[Dict]: List of dictionary where each item has two key item: 
            (a) 'class' : the class id associated to the region, 
            (b) 'region' : the extracted isolated region. The region blob is binalized so the value is {0,1}.
    """
    # Determine the number of class labels
    labels = np.unique(data.flatten()).tolist()
    if len(labels) < 2:
        return None

    result = []
    for i in range(1, len(labels)):
        clss_id = labels[i]
        mask = data == clss_id
        class_layer = data * mask

        numLabels, area, _, _ = cv2.connectedComponentsWithStats(class_layer, 4)
        for j in range(1, numLabels):
            mask = area == j
            region = data * mask
            result.append({
                'class' : clss_id,
                'region' : region
            })

    return result

def iou_binary(prediction : np.ndarray, target : np.ndarray):
    """Measuring mean IoU metric for binary images

    Args:
        prediction (np.ndarray): The image containing the prediction
        target (np.ndarray): The image containing the target

    Returns:
        float: the mean of IoU across the IoU of all regions
    """

    # Calculate intersection
    intersection = np.count_nonzero(np.logical_and(prediction, target))
    # Calculate union
    union = np.count_nonzero(np.logical_or(prediction, target))
    
    return float(intersection) / float(union)


def phm_mIoU(
    prediction : np.ndarray, 
    target : np.ndarray, 
    iou_thresh : float = 0, 
    details : bool = False
):
    """ Measuring mean IoU

    Args:
        prediction (np.ndarray): The image containing the prediction
        target (np.ndarray): The image containing the target
        iou_thresh (float, optional): The threshold to filter out IoU measures. Defaults to 0.
        details (bool, optional): Determines whether the function's return contains the detailed results or not! Defaults to False.

    Returns:
        float: mean IoU
        numpy.ndarray : the table containing the IoU values for each region in target and prediction.
    """
    pregs = extract_regions(prediction)
    tregs = extract_regions(target)
    # Extract regions from the result
    pregs = [p['region'] for p in pregs]
    tregs = [t['region'] for t in tregs]
    
    iou_map = np.zeros((len(pregs), len(tregs)))
    for pid in range(len(pregs)):
        p = pregs[pid]        
        pt = p > 0
        for tid in range(len(tregs)):
            t = tregs[tid]
            tt = t > 0
            iou_map[pid,tid] = iou_binary(pt,tt)
    
    max_iou = np.amax(iou_map, axis=1)
    iou = np.mean(max_iou[max_iou > iou_thresh])

    if details:
        return iou, iou_map

    return iou

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
    for i in range(org.shape[2]):
        dif = np.subtract(org[:, :, i], pred[:, :, i])
        m = np.mean(np.square(dif / max_p))
        s = np.sqrt(m)
        rmse_bands.append(s)

    return np.mean(rmse_bands)

class Metric (object):

    def __init__(self, name) -> None:
        self.__name = 'metric'
    
    def __call__(self, src, dsc):
        return self._process(src, dsc)

    @abstractmethod
    def _process(src, dsc):
        pass

    def __str__(self) -> str:
        return self._name
     