"""Geometry-related modules for the Repairs environment.

This package contains all geometry-related code, including tool configurations,
collision geometries, and spatial transformations.

This module provides both MuJoCo and Genesis-compatible implementations.
"""

# Genesis-compatible implementations
from .genesis_geometry import GenesisWorld, create_default_genesis_world
from .genesis_car_components import Battery, EngineBay, CarFront, create_car_components

# Re-export for backward compatibility
from .car_components import *  # noqa
from .world_geometry import *  # noqa
from .config import *  # noqa

# Explicitly list all public API symbols
__all__ = [
    # Genesis components
    'GenesisWorld',
    'create_default_genesis_world',
    'Battery',
    'EngineBay',
    'CarFront',
    'create_car_components'
]

# Also include all non-underscored symbols from other modules
__all__.extend([name for name in dir() if not name.startswith('_') and name not in __all__])
