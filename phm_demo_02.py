
from phm import PHMAutoencoder01Segmentator, load_config, Wonjik2020Segmentator, PHMAutoencoder01, Kanezaki2018Segmentator

import os
import cv2

# config = load_config('phm01_config.json')
config = load_config('kanezaki2020_config.json')

test_dir = 'datasets'
test_filename = 'aerospace_defect_crop_enhanced.jpg' # '8049.jpg' # 'pipe_color_plate.jpg'

model = PHMAutoencoder01(config['model'], 3)
segs = Wonjik2020Segmentator(config, model=model)
# segs = Kanezaki2018Segmentator(config, model=model)
# segs = PHMAutoencoder01Segmentator(config)

img = cv2.imread(os.path.join(test_dir,test_filename))
if img is None:
    raise ValueError('The image is invalid!')

a = segs.segment(img)
print(a)
