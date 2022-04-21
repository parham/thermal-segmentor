
""" 
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University

    @year           2018
    @repo           https://github.com/parham/thermal-segmentor
"""

from .core import *
from .kanezaki2018 import *
from .wonjik2020 import *
from .phm2022_autoencoder import *
from .classics import *
from .segment import *

# Check CUDA Availability
import torch, gc

if not torch.cuda.is_available():
    print('CUDA is not available!')
else:
    gc.collect()
    torch.cuda.empty_cache()

# Initialize the folders
from pathlib import Path
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./datasets").mkdir(parents=True, exist_ok=True)

# Initialize the logging
initialize_log()