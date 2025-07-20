# Configuration modules for neuraloperator
from . import default_config
from . import models
from . import opt
from . import wandb
from . import distributed

# Import specific configs for different problems
from . import darcy_config
from . import burgers_config
from . import navier_stokes_config
from . import uqno_config

__all__ = [
    'default_config',
    'models', 
    'opt',
    'wandb',
    'distributed',
    'darcy_config',
    'burgers_config', 
    'navier_stokes_config',
    'uqno_config',
]