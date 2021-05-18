
from phm import Wonjik2020Segmentator, load_config, Wonjik2020Segmentator_, Wonjik2020Net

import os
import cv2

import torch.optim as optim

config = load_config('wonjik2020_config.json')

test_dir = 'datasets'
test_filename = 'aerospace_defect_crop_enhanced.jpg'

segs = Wonjik2020Segmentator_(config)

# segs = Wonjik2020Segmentator(config)

img = cv2.imread(os.path.join(test_dir,test_filename))
if img is None:
    raise ValueError('The image is invalid!')

segs.segment(img)

