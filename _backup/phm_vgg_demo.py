
from phm import load_config, \
    VGGBasedAutoEncoder, UnsupervisedSegmentor

import os
import cv2

config = load_config('phm_vgg_config.json')

test_dir = 'datasets'
test_filename = 'aerospace_defect_crop_enhanced.jpg' # '8049.jpg' # 'pipe_color_plate.jpg'

model = VGGBasedAutoEncoder(config['model'], 3)
segs = UnsupervisedSegmentor(config, model=model)

img = cv2.imread(os.path.join(test_dir,test_filename))
img = cv2.resize(img, (512, 512))
if img is None:
    raise ValueError('The image is invalid!')

a = segs.segment(img)
print(a)
