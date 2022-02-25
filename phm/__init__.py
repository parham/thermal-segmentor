
""" 
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University

    @year           2018
    @repo           https://github.com/parham/thermal-segmentor
"""


from .core import *
from ..phm2.segment import *
from ..phm2.wonjik import *
from ..phm2.kanezaki import *
from ..phm2.phm_autoenc import *
from ..phm2.phm_vgg import *

# Check CUDA Availability
if not torch.cuda.is_available():
    print(terminal('CUDA is not available!', TerminalStyle.WARNING))

# Initialize the folders
from pathlib import Path
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./datasets").mkdir(parents=True, exist_ok=True)

# Initialize the logging
initialize_log()