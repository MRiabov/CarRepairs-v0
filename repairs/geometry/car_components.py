"""Car components for the Repairs environment.

This module defines the 3D models and their specifications for various car components
using JAX-compatible operations for efficient simulation.

Components have randomized dimensions and positions within realistic ranges to create
diverse repair scenarios.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, TypeVar, Tuple

import jax.numpy as jp
from jax import random
import numpy as np

try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False

# Type aliases
JaxArray = jp.ndarray
NpArray = np.ndarray
Array = TypeVar('Array', JaxArray, NpArray)  # Type that can be either JAX or NumPy array

# Randomization parameters
RANDOM_SCALE = 0.2  # ±20% variation in dimensions
MIN_COMPONENT_DISTANCE = 0.1  # Minimum distance between components (meters)
MIN_ENGINE_HEIGHT = 0.2  # Minimum height for engine bay placement
MAX_ATTEMPTS = 10  # Max attempts to find non-overlapping positions

@dataclass
class Battery:
    """Battery system component with randomized dimensions.
    
    Attributes:
        key: JAX random key for reproducibility
        pos: Position of the battery center in world coordinates [x, y, z]
        euler: Orientation of the battery as euler angles [roll, pitch, yaw] in radians
        dims: Dimensions of the battery [length, width, height] in meters
        mass: Mass of the battery in kg
    """
    key: jp.ndarray
    pos: jp.ndarray = field(init=False)
    euler: jp.ndarray = field(default_factory=lambda: jp.array([0., 0., 0.]))
    dims: jp.ndarray = field(init=False)
    mass: float = field(init=False)
    
    def __post_init__(self):
        """Initialize battery with random dimensions and mass."""
        # Generate random dimensions within ±20% of nominal
        key1, key2 = random.split(self.key, 2)
        base_dims = jp.array([0.30, 0.17, 0.19])  # Nominal dimensions
        scale = 1.0 + RANDOM_SCALE * (2.0 * random.uniform(key1, (3,)) - 1.0)
        self.dims = base_dims * scale
        
        # Mass scales with volume (approximate)
        base_mass = 15.0  # kg for nominal size
        volume_scale = jp.prod(scale)
        self.mass = base_mass * volume_scale
        
        # Random position will be set by the parent component
        self.pos = jp.array([0., 0., 0.])
    
    def create_mjcf(self, parent) -> Dict[str, Any]:
        """Create MJCF elements for the battery system.
        
        Args:
            parent: The parent MJCF element to attach the battery to.
            
        Returns:
            Dictionary containing the created MJCF elements.
        """
        # Main battery body
        battery_body = parent.worldbody.add('body', name='battery', pos=self.pos, euler=self.euler)
        
        # Battery main body
        battery_geom = battery_body.add('geom', 
                                      type='box', 
                                      size=self.dims/2,  # MJCF uses half-extents
                                      rgba=[0.9, 0.9, 0.9, 1.0],  # Light gray
                                      mass=self.mass,
                                      name='battery_body')
        
        # Battery terminals (positive and negative)
        terminal_radius = 0.01  # 1cm radius
        terminal_height = 0.03  # 3cm height
        
        # Position terminals on top of the battery
        terminal_offset = jp.array([0.08, 0, self.dims[2]/2 + terminal_height/2])
        
        # Positive terminal (red)
        battery_body.add('geom', 
                       type='cylinder', 
                       size=[terminal_radius, terminal_height/2],
                       pos=terminal_offset * jp.array([1, 1, 1]),
                       euler=[0, jp.pi/2, 0],
                       rgba=[0.8, 0.1, 0.1, 1.0],  # Red
                       name='positive_terminal')
        
        # Negative terminal (black)
        battery_body.add('geom', 
                       type='cylinder', 
                       size=[terminal_radius, terminal_height/2],
                       pos=terminal_offset * jp.array([1, -1, 1]),
                       euler=[0, jp.pi/2, 0],
                       rgba=[0.1, 0.1, 0.1, 1.0],  # Black
                       name='negative_terminal')
        
        return {
            'battery_body': battery_geom,
            'battery_terminals': ['positive_terminal', 'negative_terminal']
        }
        
    def create_genesis_entity(self, scene: Any, name: str = "battery") -> Dict[str, Any]:
        """Create a Genesis entity for this battery.
        
        Args:
            scene: The Genesis scene to add the battery to.
            name: Name for the battery entity.
            
        Returns:
            Dictionary containing the created entities.
        """
        if not GENESIS_AVAILABLE:
            raise ImportError("Genesis is not available. Please install the genesis package.")
            
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
        
        # Create terminals
        terminal_radius = 0.01
        terminal_height = 0.03
        terminal_offset = np.array([0.08, 0, self.dims[2]/2 + terminal_height/2])
        
        # Positive terminal (red)
        pos_terminal = scene.add_entity(
            gs.morphs.Cylinder(radius=terminal_radius, height=terminal_height),
            surface=gs.surfaces.Plastic(color=(0.8, 0.1, 0.1, 1.0))
        )
        pos_terminal.position = [float(x) for x in (self.pos + terminal_offset * np.array([1, 1, 1]))]
        pos_terminal.euler = [0, np.pi/2, 0]
        pos_terminal.name = f"{name}_positive_terminal"
        
        # Negative terminal (black)
        neg_terminal = scene.add_entity(
            gs.morphs.Cylinder(radius=terminal_radius, height=terminal_height),
            surface=gs.surfaces.Plastic(color=(0.1, 0.1, 0.1, 1.0))
        )
        neg_terminal.position = [float(x) for x in (self.pos + terminal_offset * np.array([1, -1, 1]))]
        neg_terminal.euler = [0, np.pi/2, 0]
        neg_terminal.name = f"{name}_negative_terminal"
        
        return {
            'battery': entity,
            'terminals': [pos_terminal, neg_terminal]
        }

@dataclass
class EngineBay:
    """Engine bay component with randomized dimensions.
    
    Attributes:
        key: JAX random key for reproducibility
        pos: Position of the engine bay center in world coordinates [x, y, z]
        euler: Orientation of the engine bay as euler angles [roll, pitch, yaw] in radians
        dims: Dimensions of the engine bay [length, width, height] in meters
        component_positions: Dictionary of component positions relative to engine bay
    """
    key: jp.ndarray
    pos: jp.ndarray = field(init=False)
    euler: jp.ndarray = field(default_factory=lambda: jp.array([0., 0., 0.]))
    dims: jp.ndarray = field(init=False)
    component_positions: Dict[str, jp.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize engine bay with random dimensions and component positions."""
        key1, key2, key3, key4 = random.split(self.key, 4)
        
        # Randomize engine bay dimensions
        base_dims = jp.array([1.2, 0.8, 0.6])  # Nominal dimensions
        scale = 1.0 + RANDOM_SCALE * (2.0 * random.uniform(key1, (3,)) - 1.0)
        self.dims = base_dims * scale
        
        # Random position will be set by the parent component
        self.pos = jp.array([0., 0., 0.])
        
        # Randomize component positions within the engine bay
        # These are relative to the engine bay position
        self.component_positions = {
            'engine_block': jp.array([0, 0, 0.15 * scale[2]]),
            'oil_fill_cap': jp.array([
                0.1 * (0.8 + 0.4 * random.uniform(key2, ())),  # x
                0.1 * (0.8 + 0.4 * random.uniform(key3, ())),  # y
                0.3 * scale[2] * (1.0 + 0.2 * random.uniform(key4, ()))  # z
            ]),
            'oil_dipstick': jp.array([
                -0.1 * (0.8 + 0.4 * random.uniform(key2, ())),  # x
                0.1 * (0.8 + 0.4 * random.uniform(key3, ())),   # y
                0.15 * scale[2]  # z
            ]),
            'oil_filter': jp.array([
                0.25 * scale[0],  # x
                0.0,  # y
                0.15 * scale[2]  # z
            ])
        }
    

    
    def create_mjcf(self, parent) -> Dict[str, Any]:
        """Create MJCF elements for the engine bay.
        
        Args:
            parent: The parent MJCF element to attach the engine bay to.
            
        Returns:
            Dictionary containing the created MJCF elements.
        """
        # Main engine bay body
        engine_bay = parent.worldbody.add('body', name='engine_bay', pos=self.pos, euler=self.euler)
        
        # Engine block (simplified as a box for now)
        engine_block_dims = self.dims * jp.array([0.33, 0.38, 0.5])  # Proportional to engine bay size
        engine_block = engine_bay.add('geom',
                                    type='box',
                                    size=engine_block_dims/2,
                                    pos=self.component_positions['engine_block'],
                                    rgba=[0.3, 0.3, 0.3, 1.0],  # Dark gray
                                    name='engine_block')
        
        # Oil fill cap (on top of the engine)
        cap_radius = 0.015 + 0.01 * random.uniform(random.PRNGKey(hash('oil_cap')), ())  # 2.5-3.5cm radius
        cap_height = 0.02
        oil_cap = engine_bay.add('geom',
                               type='cylinder',
                               size=[cap_radius, cap_height/2],
                               pos=self.component_positions['oil_fill_cap'],
                               euler=[jp.pi/2, 0, 0],
                               rgba=[0.8, 0.8, 0.0, 1.0],  # Yellow
                               name='oil_fill_cap')
        
        # Oil dipstick (simplified as a small cylinder)
        dipstick_radius = 0.004 + 0.002 * random.uniform(random.PRNGKey(hash('dipstick')), ())  # 4-6mm radius
        dipstick_length = 0.15 + 0.1 * random.uniform(random.PRNGKey(hash('dipstick_len')), ())  # 15-25cm length
        dipstick = engine_bay.add('geom',
                                type='cylinder',
                                size=[dipstick_radius, dipstick_length/2],
                                pos=self.component_positions['oil_dipstick'],
                                euler=[0, jp.pi/2, 0],
                                rgba=[0.9, 0.9, 0.9, 1.0],  # Light gray
                                name='oil_dipstick')
        
        # Oil filter (cylinder on the side of the engine)
        filter_radius = 0.035 + 0.01 * random.uniform(random.PRNGKey(hash('filter')), ())  # 7-9cm diameter
        filter_height = 0.08 + 0.04 * random.uniform(random.PRNGKey(hash('filter_len')), ())  # 8-12cm height
        oil_filter = engine_bay.add('geom',
                                  type='cylinder',
                                  size=[filter_radius, filter_height/2],
                                  pos=self.component_positions['oil_filter'],
                                  euler=[0, jp.pi/2, 0],
                                  rgba=[0.8, 0.8, 0.8, 1.0],  # Light gray
                                  name='oil_filter')
        
        return {
            'engine_block': engine_block,
            'oil_fill_cap': oil_cap,
            'oil_dipstick': dipstick,
            'oil_filter': oil_filter
        }
        
    def create_genesis_entities(self, scene: Any, parent_name: str = "car") -> Dict[str, Any]:
        """Create Genesis entities for the engine bay.
        
        Args:
            scene: The Genesis scene to add the engine bay to.
            parent_name: Name of the parent entity.
            
        Returns:
            Dictionary containing the created entities.
        """
        if not GENESIS_AVAILABLE:
            raise ImportError("Genesis is not available. Please install the genesis package.")
            
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
        
        # Create engine block
        engine_block_dims = self.dims * np.array([0.33, 0.38, 0.5])
        engine_block = scene.add_entity(
            gs.morphs.Box(half_extents=engine_block_dims/2),
            surface=gs.surfaces.Plastic(color=(0.2, 0.2, 0.2, 1.0))
        )
        engine_block.position = [float(x) for x in (self.pos + self.component_positions['engine_block'])]
        engine_block.name = f"{parent_name}_engine_block"
        entities['engine_block'] = engine_block
        
        # Create oil fill cap
        cap_radius = 0.025  # 2.5cm radius
        cap_height = 0.02
        oil_cap = scene.add_entity(
            gs.morphs.Cylinder(radius=cap_radius, height=cap_height),
            surface=gs.surfaces.Plastic(color=(0.8, 0.8, 0.0, 1.0))
        )
        oil_cap.position = [float(x) for x in (self.pos + self.component_positions['oil_fill_cap'])]
        oil_cap.euler = [np.pi/2, 0, 0]  # Rotate to be horizontal
        oil_cap.name = f"{parent_name}_oil_fill_cap"
        entities['oil_fill_cap'] = oil_cap
        
        # Create oil dipstick
        dipstick_radius = 0.005  # 5mm radius
        dipstick_length = 0.2  # 20cm length
        dipstick = scene.add_entity(
            gs.morphs.Cylinder(radius=dipstick_radius, height=dipstick_length),
            surface=gs.surfaces.Plastic(color=(0.9, 0.9, 0.9, 1.0))
        )
        dipstick.position = [float(x) for x in (self.pos + self.component_positions['oil_dipstick'])]
        dipstick.euler = [0, np.pi/2, 0]  # Rotate to be horizontal
        dipstick.name = f"{parent_name}_oil_dipstick"
        entities['oil_dipstick'] = dipstick
        
        return entities

@dataclass
class CarFront:
    """Front section of a car including engine bay and surrounding components.
    
    Attributes:
        key: JAX random key for reproducibility
        pos: Position of the car front center in world coordinates [x, y, z]
        euler: Orientation of the car front as euler angles [roll, pitch, yaw] in radians
    """
    key: jp.ndarray
    pos: jp.ndarray = field(default_factory=lambda: jp.array([0.0, 0.0, 0.0]))
    euler: jp.ndarray = field(default_factory=lambda: jp.array([0.0, 0.0, 0.0]))
    
    def __post_init__(self):
        """Initialize car front with components."""
        key1, key2, key3 = random.split(self.key, 3)
        
        # Create components with random properties
        self.engine_bay = EngineBay(key=key1)
        self.battery = Battery(key=key2)
        
        # Position components
        self._position_components(key3)
    
    def _position_components(self, key: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
        """Position components within the car front.
        
        Args:
            key: JAX random key for reproducibility
            
        Returns:
            Tuple containing the positions of (engine_bay, battery, car_front)
        """
        # Place engine bay in the front of the car
        engine_pos = jp.array([
            1.2 + 0.2 * (2.0 * random.uniform(key, ()) - 1.0),  # x: 1.0-1.4
            0.0,  # y: centered
            0.4 + 0.1 * random.uniform(key, ())  # z: 0.4-0.5
        ])
        
        # Place battery near the engine bay but not overlapping
        battery_pos = jp.array([0.0, 0.0, 0.0])
        key, subkey = random.split(key)
        
        for _ in range(10):  # MAX_ATTEMPTS
            key, subkey = random.split(key)
            # Try placing battery either left or right of engine bay
            side = 1.0 if random.uniform(subkey, ()) > 0.5 else -1.0
            battery_pos = jp.array([
                0.8 + 0.2 * (2.0 * random.uniform(subkey, ()) - 1.0),  # x: 0.6-1.0
                side * (0.3 + 0.1 * random.uniform(subkey, ())),  # y: ±(0.3-0.4)
                0.2 + 0.1 * random.uniform(subkey, ())  # z: 0.2-0.3
            ])
            
            # Check for collision with engine bay
            if jp.linalg.norm(battery_pos - engine_pos) > 0.5:  # Minimum distance
                break
        else:
            # If we couldn't find a non-overlapping position, place it at a safe distance
            battery_pos = jp.array([
                0.8,
                0.5 * (1.0 if random.uniform(key, ()) > 0.5 else -1.0),
                0.25
            ])
        
        # Set component positions
        self.engine_bay.pos = engine_pos
        self.battery.pos = battery_pos
        
        # Calculate car front position (centered between engine bay and battery)
        car_front_pos = (engine_pos + battery_pos) / 2.0
        max_z = float(jp.maximum(engine_pos[2], battery_pos[2]))
        car_front_pos = car_front_pos.at[2].set(max_z + 0.1)
        
        return engine_pos, battery_pos, car_front_pos
    
    def create_mjcf(self, parent) -> Dict[str, Any]:
        """Create MJCF elements for the car front.
        
        Args:
            parent: The parent MJCF element to attach the car front to.
            
        Returns:
            Dictionary containing the created MJCF elements.
        """
        car_body = parent.worldbody.add('body', name='car_front', pos=self.pos, euler=self.euler)
        
        # Add car body (simplified front section)
        body_dims = jp.array([2.5, 1.0, 0.5])  # Length, width, height
        car_body.add('geom',
                    type='box',
                    size=body_dims/2,
                    pos=jp.array([1.0, 0.0, 0.25]),
                    rgba=[0.2, 0.2, 0.8, 1.0],  # Blue
                    name='car_body')
        
        # Add front bumper
        bumper_dims = jp.array([0.1, 1.2, 0.2])
        car_body.add('geom',
                    type='box',
                    size=bumper_dims/2,
                    pos=jp.array([2.4, 0.0, 0.1]),
                    rgba=[0.1, 0.1, 0.1, 1.0],  # Black
                    name='front_bumper')
        
        # Add components
        components = {}
        components['battery'] = self.battery.create_mjcf(car_body)
        components['engine_bay'] = self.engine_bay.create_mjcf(car_body)
        
        return components

def create_car_components(key: jp.ndarray) -> Tuple[Battery, EngineBay]:
    """Create and return the car components with randomized positions and sizes.
    
    This function is deprecated. Use the CarFront class instead for better
    component placement and collision avoidance.
    
    Args:
        key: JAX random key for reproducibility
        
    Returns:
        A tuple containing (battery, engine_bay) components with randomized properties.
    """
    # Create a CarFront instance which handles component creation and placement
    car_front = CarFront(key=key)
    return car_front.battery, car_front.engine_bay
