"""
RESA Add-ons package.

Contains optional modules for extended functionality:
- tank: Tank pressure simulation for bi-propellant systems
- injector: Injector design and analysis
- igniter: Torch igniter sizing tool for Ethanol/N2O igniters
- contour: Nozzle contour generation
"""

from . import tank
from . import igniter

__all__ = [
    'tank',
    'igniter',
]
