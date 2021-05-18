
from phm import PHMAutoencoder01Segmentator, load_config

import os
import cv2

config = load_config('phm01_config.json')

test_dir = 'datasets'
test_filename = 'aerospace_defect_crop_enhanced.jpg' # '8049.jpg' # 'pipe_color_plate.jpg'

segs = PHMAutoencoder01Segmentator(config)

img = cv2.imread(os.path.join(test_dir,test_filename))
if img is None:
    raise ValueError('The image is invalid!')

a = segs.segment(img)
print(a)
