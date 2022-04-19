
import os
import logging
import numpy as np

from PIL import Image
from datetime import datetime
from comet_ml import Experiment

from phm.kanezaki2018 import create_noref_predict_Kanezaki2018__

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
experiment.set_name('%s_%s_%s' % ('kanezaki2018', now.strftime('%Y%m%d-%H%M'), sample_file.split('.')[0]))
experiment.add_tag(sample_file.split('.')[0])

engine = create_noref_predict_Kanezaki2018__(
    config_file='configs/kanezaki2018.json',
    experiment=experiment,
    metrics=None
)

img = Image.open(os.path.join(root_dir,sample_file))
img = img.convert('RGB')
img = np.asarray(img)

if img is None:
    err_msg = f'{sample_file} is not valid file!'
    logging.error(err_msg)
    raise ValueError(err_msg)

engine.run([[img,np.zeros(img.shape)]])

experiment.end()
