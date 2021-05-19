

from .core import *
from .segment import *
from .wonjik import *
from .kanezaki import *
from .phm_autoenc import *
from .phm_vgg import *

# Initialize the folders
from pathlib import Path
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./datasets").mkdir(parents=True, exist_ok=True)

# Initialize the logging
initialize_log()