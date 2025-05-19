"""
Genesis-compatible world geometry for the repair simulation.

This module provides a Genesis-based implementation of the world geometry,
including the workbench, car components, and tool stand.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import jax
import jax.numpy as jp
from jax import random
import genesis as gs
import numpy as np

from .car_components import CarFront
from .config import TOOL_STAND_POS, TOOL_STAND_DIMS, TOOL_SLOT_SPACING, Array

# Constants for world layout
WORKBENCH_DIMS = jp.array([3.0, 2.0, 0.1])  # Length, width, height (meters)
FLOOR_DIMS = jp.array([10.0, 10.0, 0.1])
ROBOT_BASE_POS = jp.array([-1.0, 0.0, 0.0])  # Position relative to workbench

@dataclass
class GenesisWorld:
    """Manages the complete geometry of the simulation world using Genesis.
    
    This class is responsible for creating and positioning all elements in the
    simulation, including the workbench, car, tools, and robotic arm.
    """
    scene: gs.Scene
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
        
        # Create all world elements
        self._create_world()
    
    def _position_car(self) -> None:
        """Position the car on the workbench."""
        # Center the car on the workbench
        car_pos = jp.array([
            self.workbench_dims[0] * 0.6,  # 60% from the front
            0.0,  # Centered left-right
            self.workbench_dims[2]  # On top of the workbench
        ])
        self.car_front.pos = car_pos
    
    def _create_world(self) -> None:
        """Create all world elements in the Genesis scene."""
        # Add floor
        self.floor = self.scene.add_entity(
            gs.morphs.Plane(size=float(self.floor_dims[0])),
            pos=(0, 0, -0.05),  # Slightly below origin
            material=gs.materials.Rigid(rgba=(0.2, 0.3, 0.2, 1.0)),
            name="floor"
        )
        
        # Add workbench
        self.workbench = self._create_workbench()
        
        # Add tool stand
        self.tool_stand = self._create_tool_stand()
        
        # Add car components
        self._create_car_components()
    
    def _create_workbench(self) -> Dict[str, Any]:
        """Create the workbench in the scene."""
        workbench = {}
        
        # Main surface
        workbench['surface'] = self.scene.add_entity(
            gs.morphs.Box(half_extents=np.array(self.workbench_dims)/2),
            pos=(0, 0, float(self.workbench_dims[2])/2),
            material=gs.materials.Rigid(rgba=(0.6, 0.6, 0.6, 1.0)),
            name="workbench_surface"
        )
        
        # Add legs
        leg_size = np.array([0.05, 0.05, 0.4])
        leg_positions = [
            [1.4, 0.9, -0.2],
            [1.4, -0.9, -0.2],
            [-1.4, 0.9, -0.2],
            [-1.4, -0.9, -0.2]
        ]
        
        workbench['legs'] = []
        for i, pos in enumerate(leg_positions):
            leg = self.scene.add_entity(
                gs.morphs.Box(half_extents=leg_size/2),
                pos=pos,
                material=gs.materials.Rigid(rgba=(0.4, 0.4, 0.4, 1.0)),
                name=f"workbench_leg_{i}"
            )
            workbench['legs'].append(leg)
        
        return workbench
    
    def _create_tool_stand(self) -> Dict[str, Any]:
        """Create the tool stand in the scene."""
        tool_stand = {}
        
        # Base of the tool stand
        tool_stand['base'] = self.scene.add_entity(
            gs.morphs.Box(half_extents=np.array(TOOL_STAND_DIMS)/2),
            pos=TOOL_STAND_POS.tolist(),
            material=gs.materials.Rigid(rgba=(0.3, 0.3, 0.3, 1.0)),
            name="tool_stand_base"
        )
        
        # Tool slots (simplified as colored cylinders)
        from .config import TOOL_GRIPPER, TOOL_SUCTION, TOOL_SCREWDRIVER, TOOL_WRENCH
        
        tool_colors = {
            TOOL_GRIPPER: (0.2, 0.8, 0.2, 1.0),    # Green
            TOOL_SUCTION: (0.8, 0.2, 0.2, 1.0),    # Red
            TOOL_SCREWDRIVER: (0.2, 0.2, 0.8, 1.0), # Blue
            TOOL_WRENCH: (0.8, 0.8, 0.2, 1.0)      # Yellow
        }
        
        tool_types = [TOOL_GRIPPER, TOOL_SUCTION, TOOL_SCREWDRIVER, TOOL_WRENCH]
        tool_stand['tools'] = {}
        
        for i, tool_type in enumerate(tool_types):
            row = i // 2
            col = i % 2
            tool_pos = [
                float(TOOL_STAND_POS[0] + (col - 0.5) * TOOL_SLOT_SPACING),
                float(TOOL_STAND_POS[1] + (0.5 - row) * TOOL_SLOT_SPACING),
                float(TOOL_STAND_POS[2] + TOOL_STAND_DIMS[2]/2 + 0.01)
            ]
            
            tool = self.scene.add_entity(
                gs.morphs.Cylinder(radius=0.05, height=0.1),
                position=tool_pos,
                material=gs.materials.Rigid(color=tool_colors[tool_type]),
                name=f"tool_{tool_type}"
            )
            tool_stand['tools'][tool_type] = tool
        
        return tool_stand
    
    def _create_car_components(self) -> None:
        """Create and position car components in the scene."""
        # Car body (simplified as a box for now)
        car_pos = self.car_front.pos.tolist()
        
        self.car_body = self.scene.add_entity(
            gs.morphs.Box(half_extents=np.array([1.0, 0.8, 0.4])/2),
            pos=car_pos,
            material=gs.materials.Rigid(rgba=(0.1, 0.1, 0.1, 1.0)),
            name="car_body"
        )
        
        # Engine bay (simplified as a box with a lid)
        engine_bay_pos = [car_pos[0], car_pos[1], car_pos[2] + 0.2]
        self.engine_bay = self.scene.add_entity(
            gs.morphs.Box(size=np.array([0.8, 0.6, 0.2]), pos=engine_bay_pos, name="engine_bay"),
            material=gs.materials.Rigid(),
            surface = gs.surfaces.Iron(),

        )
        
        # Engine bay lid (slightly open)
        self.engine_bay_lid = self.scene.add_entity(
            gs.morphs.Box(size=np.array([0.8, 0.6, 0.02]), pos=[engine_bay_pos[0], engine_bay_pos[1], engine_bay_pos[2] + 0.2], euler=[0.2, 0, 0]),#slightly open
            surface=gs.surfaces.Aluminium(),
        )

def create_default_genesis_world(scene: gs.Scene, key: Optional[jp.ndarray] = None) -> GenesisWorld:
    """Create a default world configuration using Genesis.
    
    Args:
        scene: The Genesis scene to add the world to.
        key: Optional JAX random key. If None, a default key will be used.
        
    Returns:
        A configured GenesisWorld instance.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    return GenesisWorld(scene=scene, key=key)
