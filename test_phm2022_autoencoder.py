
import logging
import os
import cv2
import numpy as np

from PIL import Image
from random import sample
from datetime import datetime
from comet_ml import Experiment

from phm import load_config
from phm import phmAutoencoderSegmentor

config = load_config('configs/phm_autoencoder.json')

root_dir = 'datasets'
sample_file = 'aerospace_defect_crop_enhanced.jpg' # 'pipe_color_plate.jpg' # 'aerospace_defect_crop_enhanced.jpg' # 'pipe_color_plate.jpg'

# Create an experiment with your api key
now = datetime.now()
experiment = Experiment(
    api_key="8CuCMLyaa23fZDEY2uTxm5THf",
    project_name="thermal-segmentor",
    workspace="parham",
    log_git_metadata=True
)
experiment.set_name('%s_%s_%s' % ('phm2022_autoencoder', now.strftime('%Y%m%d-%H%M'), sample_file.split('.')[0]))
experiment.add_tag(sample_file.split('.')[0])
experiment.log_parameters(config.model, prefix='model')
experiment.log_parameters(config.segmentation, prefix='segmentation')

segs = phmAutoencoderSegmentor(config, experiment=experiment, use_cuda=False)

img = Image.open(os.path.join(root_dir,sample_file))
img = img.convert('RGB')
img = np.asarray(img)

if img is None:
    err_msg = f'{sample_file} is not valid file!'
    logging.error(err_msg)
    raise ValueError(err_msg)

res = segs.segment(img)
print(res)
experiment.end()
