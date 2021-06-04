
from phm import Kanezaki2018Segmentator, load_config

import os
import cv2
import PIL as pil
import numpy as np

config = load_config('kanezaki2018_config.json')

test_dir = 'datasets'
test_filename = 'aerospace_defect_crop_enhanced.jpg' # 'pipe_color_plate.jpg' # 'aerospace_defect_crop_enhanced.jpg' # 'pipe_color_plate.jpg'

segs = Kanezaki2018Segmentator(config)

img = pil.Image.open(os.path.join(test_dir,test_filename))
img = img.convert('RGB')
img = np.asarray(img)

if img is None:
    raise ValueError('The image is invalid!')

segs.segment(img)
