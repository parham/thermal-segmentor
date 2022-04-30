
""" 
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import numpy as np
from typing import List

from phm.metrics import extract_regions


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
