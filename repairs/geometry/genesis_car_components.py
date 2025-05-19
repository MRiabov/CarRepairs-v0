"""
Genesis-compatible car components for the Repairs environment.

This module defines the 3D models and their specifications for various car components
using JAX-compatible operations for efficient simulation with Genesis.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Union

import jax.numpy as jp
from jax import random
import genesis as gs
import numpy as np

# Type aliases
JaxArray = jp.ndarray
NpArray = np.ndarray
Array = Union[JaxArray, NpArray]  # Type that can be either JAX or NumPy array

# Randomization parameters
RANDOM_SCALE = 0.2  # ±20% variation in dimensions
MIN_COMPONENT_DISTANCE = 0.1  # Minimum distance between components (meters)
MIN_ENGINE_HEIGHT = 0.2  # Minimum height for engine bay placement
MAX_ATTEMPTS = 10  # Max attempts to find non-overlapping positions

@dataclass
class Battery:
    """Battery system component with randomized dimensions."""
    key: JaxArray
    pos: JaxArray = field(init=False)
    euler: JaxArray = field(default_factory=lambda: jp.array([0., 0., 0.]))
    dims: JaxArray = field(init=False)
    mass: float = field(init=False)
    
    def __post_init__(self):
        """Initialize battery with random dimensions and mass."""
        # Generate random dimensions within ±20% of nominal
        key1, key2 = random.split(self.key, 2)
        base_dims = jp.array([0.30, 0.17, 0.19])  # Nominal dimensions
        scale = 1.0 + RANDOM_SCALE * (2.0 * random.uniform(key1, (3,)) - 1.0)
        self.dims = base_dims * scale
        
        # Mass scales with volume (approximate)
        volume = float(jp.prod(self.dims))
        self.mass = 15.0 * volume  # Approximate density for car battery
        
        # Position will be set by parent component
        self.pos = jp.zeros(3)
    
    def create_entity(self, scene: gs.Scene, name: str = "battery") -> Any:
        """Create a Genesis entity for this battery."""
        # Create a box morph for the battery
        box = gs.morphs.Box(half_extents=np.array(self.dims)/2)
        
        # Create the entity with visual material
        entity = scene.add_entity(
            box,
            surface=gs.surfaces.Plastic(color=(0.2, 0.2, 0.2, 1.0))
        )
        
        # Set position and orientation
        entity.position = [float(x) for x in self.pos]
        entity.euler = [float(x) for x in self.euler]
        entity.name = name
        
        return entity

@dataclass
class EngineBay:
    """Engine bay component with randomized dimensions."""
    key: jp.ndarray
    pos: jp.ndarray = field(init=False)
    euler: jp.ndarray = field(default_factory=lambda: jp.array([0., 0., 0.]))
    dims: jp.ndarray = field(init=False)
    component_positions: Dict[str, jp.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize engine bay with random dimensions and component positions."""
        key1, key2 = random.split(self.key)
        
        # Randomize engine bay dimensions
        base_dims = jp.array([0.8, 0.6, 0.4])
        scale = 1.0 + RANDOM_SCALE * (2.0 * random.uniform(key1, (3,)) - 1.0)
        self.dims = base_dims * scale
        
        # Position will be set by parent component
        self.pos = jp.zeros(3)
        
        # Initialize component positions
        self._position_components(key2)
    
    def _position_components(self, key: jp.ndarray) -> None:
        """Randomly position components within the engine bay."""
        key1, key2 = random.split(key)
        
        # Create battery
        self.battery = Battery(key=key1)
        
        # Position battery in the engine bay (simplified for now)
        self.battery.pos = jp.array([
            -self.dims[0] * 0.3,  # Towards the front
            0.0,  # Centered left-right
            self.dims[2] * 0.3  # Above the bottom
        ])
        
        # Store component positions
        self.component_positions = {
            'battery': self.battery.pos
        }
    
    def create_entities(self, scene: gs.Scene, parent_name: str = "car") -> Dict[str, Any]:
        """Create Genesis entities for the engine bay and its components."""
        entities = {}
        
        # Create engine bay body
        box = gs.morphs.Box(half_extents=np.array(self.dims)/2)
        
        # Create the entity with visual material
        engine_bay = scene.add_entity(
            box,
            surface=gs.surfaces.Plastic(color=(0.3, 0.3, 0.3, 1.0))
        )
        engine_bay.position = [float(x) for x in self.pos]
        engine_bay.euler = [float(x) for x in self.euler]
        engine_bay.name = f"{parent_name}_engine_bay"
        entities['engine_bay'] = engine_bay
        
        # Create components
        battery_entity = self.battery.create_entity(
            scene, 
            name=f"{parent_name}_battery"
        )
        entities['battery'] = battery_entity
        
        return entities

@dataclass
class CarFront:
    """Front section of a car including engine bay and surrounding components."""
    key: jp.ndarray
    pos: jp.ndarray = field(default_factory=lambda: jp.array([0.0, 0.0, 0.0]))
    euler: jp.ndarray = field(default_factory=lambda: jp.array([0.0, 0.0, 0.0]))
    
    def __post_init__(self):
        """Initialize car front with components."""
        key1, key2 = random.split(self.key)
        
        # Create engine bay
        self.engine_bay = EngineBay(key=key1)
    
    def create_entities(self, scene: gs.Scene) -> Dict[str, Any]:
        """Create Genesis entities for the car front and its components."""
        entities = {}
        
        # Create car body
        box = gs.morphs.Box(half_extents=np.array([1.0, 0.8, 0.4])/2)
        
        # Create the entity with visual material
        car_body = scene.add_entity(
            box,
            surface=gs.surfaces.Plastic(color=(0.1, 0.1, 0.1, 1.0))
        )
        car_body.position = [float(x) for x in self.pos]
        car_body.euler = [float(x) for x in self.euler]
        car_body.name = "car_body"
        entities['body'] = car_body
        
        # Create engine bay and components
        engine_entities = self.engine_bay.create_entities(scene, "car")
        entities.update(engine_entities)
        
        return entities

def create_car_components(key: jp.ndarray) -> Tuple[Any, Any]:
    """Create and return the car components with randomized positions and sizes.
    
    Args:
        key: JAX random key for reproducibility
        
    Returns:
        A tuple containing (battery, engine_bay) components with randomized properties.
    """
    key1, key2 = random.split(key)
    
    # Create engine bay with battery
    engine_bay = EngineBay(key=key1)
    
    return engine_bay.battery, engine_bay
