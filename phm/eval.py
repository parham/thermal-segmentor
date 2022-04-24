
import cv2
import numpy as np
from random import choice
from typing import Dict, List

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
    # Calculate IoU
    iou = float(intersection) / float(union) if union != 0 else 0
    return iou

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
            if np.sum(region) > 0:
                result.append({
                    'class' : clss_id,
                    'region' : region
                })
            else:
                print('yee')

    return result

def adapt_output(
    output : np.ndarray,
    target : np.ndarray,
    iou_thresh : float = 0.1,
    use_null_class : bool = False):
    # a. Extract Regions
    p_regs = extract_regions(output)
    t_regs = extract_regions(target)
    p_regs = [p['region'] for p in p_regs]
    t_regs = [t['region'] for t in t_regs]
    # b. Calculate the IoU map of prediction-region map
    # b.1. Create a matrix n_p x n_t (M) ... rows are predictions and columns are targets
    p_count = len(p_regs)
    t_count = len(t_regs)
    iou_map = np.zeros((p_count, t_count))
    for pid in range(p_count):
        p = p_regs[pid]
        p_bin = p > 0
        for tid in range(t_count):
            t = t_regs[tid]
            t_bin = t > 0
            iou_map[pid,tid] = iou_binary(p_bin, t_bin) 

    labels = np.unique(target).tolist()
    null_class = choice([i for i in range(np.max(labels)) if i not in labels])
    maxv = np.amax(iou_map, axis=1).tolist()
    selected_index = np.argmax(iou_map, axis=1).tolist()
    result = np.zeros(p_regs[0].shape, dtype=np.uint8)
    coupled = []
    for i in range(len(selected_index)):
        mv = maxv[i]
        # if mv > iou_thresh:
        preg = p_regs[i]
        treg = t_regs[selected_index[i]]
        classid = np.unique(treg).tolist()
        if len(classid) > 1:
            classid = classid[-1]
            if use_null_class:
                result[preg > 0] = classid if mv > iou_thresh else null_class
                coupled.append((preg, treg))
            elif mv > iou_thresh:
                result[preg > 0] = classid
                coupled.append((preg, treg))
    
    return result, iou_map, coupled
