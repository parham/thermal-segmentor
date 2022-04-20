
from typing import Dict, List
import cv2
import numpy as np

def extract_regions(data : np.ndarray) -> List[Dict]:
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