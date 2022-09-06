
""" 
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University

    @year           2018
    @repo           https://github.com/parham/thermal-segmentor
"""

from .core import *
from .loss import *
from .models import *
from .segmentation import *

# Check CUDA Availability
import torch, gc

if not torch.cuda.is_available():
    logging.warning('CUDA is not available')
else:
    # Clearing the GPU memory
    gc.collect()
    torch.cuda.empty_cache()

# Initialize the folders
from pathlib import Path
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./datasets").mkdir(parents=True, exist_ok=True)

# Initialize the logging
initialize_log()