
from phm import load_config, \
    Wonjik2020Segmentator, Kanezaki2018Segmentator, \
    UnsupervisedNN, UnsupervisedSegmentor

import os
import cv2

config = load_config('phm01_config.json')

test_dir = 'datasets'
test_filename = 'pipe_color_plate.jpg' # 'aerospace_defect_crop_enhanced.jpg' # '8049.jpg' # 'pipe_color_plate.jpg'

model = UnsupervisedNN(config['model'], 3)
segs = UnsupervisedSegmentor(config, model=model)
# segs = Kanezaki2018Segmentator(config, model=model)

img = cv2.imread(os.path.join(test_dir,test_filename))
if img is None:
    raise ValueError('The image is invalid!')

a = segs.segment(img)
print(a)
