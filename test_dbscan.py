

import os
import numpy as np

from PIL import Image
from datetime import datetime

from comet_ml import Experiment
from phm.dbscan import DBSCAN_Impl


root_dir = 'datasets'
sample_file = 'pipe_color_plate.jpg' # 'pipe_color_plate.jpg' # 'aerospace_defect_crop_enhanced.jpg' # 'pipe_color_plate.jpg'

img = Image.open(os.path.join(root_dir,sample_file))
img = img.convert('RGB')
img = np.asarray(img)

# Create an experiment with your api key
now = datetime.now()

experiment = Experiment(
    api_key="8CuCMLyaa23fZDEY2uTxm5THf",
    project_name="thermal-segmentor",
    workspace="parham",
    log_git_metadata=True
)
experiment.set_name('%s_%s_%s' % ('dbscan', now.strftime('%Y%m%d-%H%M'), sample_file.split('.')[0]))
experiment.add_tag(sample_file.split('.')[0])

obj = DBSCAN_Impl(experiment=experiment)
obj.segment_noref(img, True, True)