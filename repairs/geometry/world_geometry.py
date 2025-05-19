"""World geometry and scene composition for the repair simulation.

This module handles the assembly of all components in the simulation world,
including the car, tools, robotic arm, and work area.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

from typing import Optional

import jax
import jax.numpy as jp
from jax import random
import mujoco
from mujoco import mjModel  # type: ignore

from repairs.geometry.car_components import CarFront
from repairs.geometry.config import TOOL_STAND_POS, TOOL_STAND_DIMS, TOOL_SLOT_SPACING

# Constants for world layout
WORKBENCH_DIMS = jp.array([3.0, 2.0, 0.1])  # Length, width, height (meters)
FLOOR_DIMS = jp.array([10.0, 10.0, 0.1])
ROBOT_BASE_POS = jp.array([-1.0, 0.0, 0.0])  # Position relative to workbench


@dataclass
class WorldGeometry:
    """Manages the complete geometry of the simulation world.
    
    This class is responsible for creating and positioning all elements in the
    simulation, including the workbench, car, tools, and robotic arm.
    
    Attributes:
        key: JAX random key for reproducibility
        workbench_dims: Dimensions of the workbench [x, y, z]
        floor_dims: Dimensions of the floor [x, y, z]
        robot_base_pos: Position of the robot base [x, y, z]
    """
    key: jp.ndarray
    workbench_dims: jp.ndarray = field(default_factory=lambda: WORKBENCH_DIMS.copy())
    floor_dims: jp.ndarray = field(default_factory=lambda: FLOOR_DIMS.copy())
    robot_base_pos: jp.ndarray = field(default_factory=lambda: ROBOT_BASE_POS.copy())
    
    def __post_init__(self):
        """Initialize the world with all components."""
        key1, key2 = random.split(self.key)
        
        # Create car front section
        self.car_front = CarFront(key=key1)
        
        # Position the car on the workbench
        self._position_car()
    
    def _position_car(self) -> None:
        """Position the car on the workbench."""
        # Center the car on the workbench
        car_pos = jp.array([
            self.workbench_dims[0] * 0.6,  # 60% from the front
            0.0,  # Centered left-right
            self.workbench_dims[2]  # On top of the workbench
        ])
        self.car_front.pos = car_pos
    
    def create_mjcf(self, model: mjModel) -> Dict[str, Any]:
        """Create MJCF elements for the entire world.
        
        Args:
            model: The MJCF model to add elements to.
            
        Returns:
            Dictionary containing references to created elements.
        """
        world = model.worldbody
        
        # Add floor
        floor = world.add('geom',
                        type='box',
                        size=self.floor_dims/2,
                        pos=jp.array([0, 0, -self.floor_dims[2]/2]),
                        rgba=[0.2, 0.3, 0.2, 1.0],
                        name='floor')
        
        # Add workbench
        workbench_pos = jp.array([0, 0, 0])
        workbench = world.add('body', name='workbench', pos=workbench_pos)
        workbench.add('geom',
                     type='box',
                     size=self.workbench_dims/2,
                     pos=jp.array([0, 0, self.workbench_dims[2]/2]),
                     rgba=[0.6, 0.6, 0.6, 1.0],
                     name='workbench_surface')
        
        # Add legs to workbench
        leg_size = jp.array([0.05, 0.05, 0.4])
        leg_positions = [
            [1.4, 0.9, -0.2],
            [1.4, -0.9, -0.2],
            [-1.4, 0.9, -0.2],
            [-1.4, -0.9, -0.2]
        ]
        
        for i, pos in enumerate(leg_positions):
            workbench.add('geom',
                        type='box',
                        size=leg_size/2,
                        pos=jp.array(pos),
                        rgba=[0.3, 0.3, 0.3, 1.0],
                        name=f'workbench_leg_{i}')
        
        # Add car to the workbench
        car_components = self.car_front.create_mjcf(workbench)
        
        # Add tool stand
        tool_stand = workbench.add('body', 
                                 name='tool_stand', 
                                 pos=TOOL_STAND_POS)
        
        tool_stand.add('geom',
                      type='box',
                      size=TOOL_STAND_DIMS/2,
                      pos=jp.array([0, 0, TOOL_STAND_DIMS[2]/2]),
                      rgba=[0.8, 0.8, 0.8, 1.0],
                      name='tool_stand_base')
        
        # Add tool slots (visual only for now)
        slot_positions = [
            [-TOOL_SLOT_SPACING, TOOL_SLOT_SPACING, 0.05],
            [0, TOOL_SLOT_SPACING, 0.05],
            [TOOL_SLOT_SPACING, TOOL_SLOT_SPACING, 0.05],
            [-TOOL_SLOT_SPACING, -TOOL_SLOT_SPACING, 0.05],
            [0, -TOOL_SLOT_SPACING, 0.05],
            [TOOL_SLOT_SPACING, -TOOL_SLOT_SPACING, 0.05]
        ]
        
        for i, pos in enumerate(slot_positions):
            tool_stand.add('geom',
                         type='cylinder',
                         size=[0.02, 0.01],  # radius, height
                         pos=jp.array(pos),
                         rgba=[0.5, 0.5, 0.5, 1.0],
                         name=f'tool_slot_{i}')
        
        # Add robot base (positioned relative to workbench)
        robot_base = world.add('body', 
                             name='robot_base', 
                             pos=workbench_pos + self.robot_base_pos)
        
        # Robot base visualization (simplified)
        robot_base.add('geom',
                      type='cylinder',
                      size=[0.15, 0.1],  # radius, height
                      pos=jp.array([0, 0, 0.1]),
                      rgba=[0.5, 0.5, 0.5, 1.0],
                      name='robot_pedestal')
        
        # Add camera rig
        camera_rig = world.add('body', name='camera_rig', pos=[0, 0, 2.5])
        
        # Camera rig geometry (invisible in rendering)
        camera_rig.add('geom',
                     type='box',
                     size=[0.05, 0.05, 0.01],
                     rgba=[0, 0, 0, 0],  # Fully transparent
                     contype=0,  # No collision
                     conaffinity=0,  # No collision
                     name='camera_rig_geom')
        
        # Camera 1: Top-down view
        camera_rig.add('camera',
                     name='camera_top',
                     pos=[0, 0, 0],
                     xyaxes=[1, 0, 0, 0, 1, 0],  # Looking straight down
                     fovy=60,
                     mode='trackcom')  # Track the center of mass of the scene
        
        # Camera 2: 45-degree angle view
        camera_rig.add('camera',
                     name='camera_angle',
                     pos=[-1.5, -1.5, 1.5],
                     xyaxes=[0.7, -0.7, 0, -0.4, -0.4, 0.8],
                     fovy=60)
        
        # Camera 3: Side view
        camera_rig.add('camera',
                     name='camera_side',
                     pos=[-2, 0, 1],
                     xyaxes=[0, 1, 0, 0.5, 0, 0.866],
                     fovy=45)
        
        # Note: The actual robot arm will be added by the robot controller
        
        return {
            'floor': floor,
            'workbench': workbench,
            'camera_rig': camera_rig,
            'tool_stand': tool_stand,
            'robot_base': robot_base,
            'car': car_components
        }

def create_default_world(key: Optional[jp.ndarray] = None) -> WorldGeometry:
    """Create a default world configuration.
    
    Args:
        key: Optional JAX random key. If None, a default key will be used.
        
    Returns:
        A configured WorldGeometry instance.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    return WorldGeometry(key=key)
