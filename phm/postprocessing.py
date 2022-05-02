
""" 
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import numpy as np
from random import choice
from typing import List

from phm.metrics import mIoU_func, extract_regions

def adapt_output(
        output: np.ndarray,
        target: np.ndarray,
        iou_thresh: float = 0.1,
        use_null_class: bool = False):

    _, iou_map, maxv, selected_index, p_regs, t_regs = mIoU_func(
        output, target, iou_thresh=iou_thresh)

    labels = np.unique(target).tolist()
    
    null_class = choice([i for i in range(np.max(labels)+10) if i not in labels])
    maxv = np.amax(iou_map, axis=1).tolist()
    selected_index = np.argmax(iou_map, axis=1).tolist()
    result = np.zeros(output.shape, dtype=np.uint8)
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

def regions_to_image(regions : List[np.ndarray]) -> np.ndarray:
    if len(regions) == 0:
        return
    res = None
    for r in regions:
        res = res + r if res is not None else r
    return res

def remove_small_regions(img : np.ndarray, min_area : int = 0):
    regs = extract_regions(img)
    regs = list(filter(lambda x : np.count_nonzero(x) > min_area, regs))
    return regions_to_image(regs)
