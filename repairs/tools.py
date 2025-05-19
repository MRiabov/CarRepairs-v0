"""Tooling system for the Repairs environment."""

from typing import Dict, Tuple, Any, List
import dataclasses
import jax.numpy as jp
from brax import base

from repairs.geometry.config import (
    TOOL_NONE, TOOL_GRIPPER, TOOL_SUCTION, TOOL_SCREWDRIVER, TOOL_WRENCH,
    TOOL_NAMES, TOOL_STAND_POS, TOOL_STAND_DIMS, TOOL_SLOT_SPACING,
    TOOL_SLOT_DIMS, MOUNT_OFFSET, Array, PyTree
)
# from brax.mjx import MjxSystem


@dataclasses.dataclass
class ToolGeometry:
    """Base class for tool geometry and properties."""
    tool_type: int
    mass: float
    dims: Array  # [width, depth, height]
    com_offset: Array  # Center of mass offset from base
    collision_geoms: List[Dict[str, Any]]
    visual_geoms: List[Dict[str, Any]]
    contact_aabb_pos: Array  # Position relative to tool tip
    contact_aabb_size: Array  # Size of the interaction AABB
    
    def get_contact_aabb(self, xform: base.Transform) -> Tuple[Array, Array]:
        """Get the world-space AABB for contact detection."""
        # Transform contact AABB to world space
        pos = xform.pos + jp.dot(xform.rot, self.contact_aabb_pos)
        half_extents = self.contact_aabb_size / 2
        
        # For now, return a simple AABB (approximate, not considering rotation)
        # In a more complete implementation, we'd compute the OBB
        return pos - half_extents, pos + half_extents


def create_gripper_geometry() -> ToolGeometry:
    """Create the parallel jaw gripper tool geometry."""
    base_dims = jp.array([0.1, 0.1, 0.15])  # Width, Depth, Height
    jaw_length = 0.08
    jaw_thickness = 0.02
    
    # Collision geometries (simplified as boxes for now)
    collision_geoms = [
        {
            'type': 'box',
            'pos': [0, 0, 0.075],  # Center of base
            'euler': [0, 0, 0],
            'size': base_dims / 2,
            'name': 'gripper_base'
        },
        {
            'type': 'box',
            'pos': [0, -0.04, 0.15 + jaw_length/2],  # Left jaw
            'euler': [0, 0, 0],
            'size': [0.02, 0.02, jaw_length/2],
            'name': 'left_jaw'
        },
        {
            'type': 'box',
            'pos': [0, 0.04, 0.15 + jaw_length/2],  # Right jaw
            'euler': [0, 0, 0],
            'size': [0.02, 0.02, jaw_length/2],
            'name': 'right_jaw'
        }
    ]
    
    # Visual geometries (same as collision for now)
    visual_geoms = collision_geoms.copy()
    
    return ToolGeometry(
        tool_type=TOOL_GRIPPER,
        mass=0.5,
        dims=base_dims,
        com_offset=jp.array([0, 0, 0.05]),
        collision_geoms=collision_geoms,
        visual_geoms=visual_geoms,
        contact_aabb_pos=jp.array([0, 0, -0.05]),
        contact_aabb_size=jp.array([0.1, 0.1, 0.1])
    )


def create_suction_geometry() -> ToolGeometry:
    """Create the suction cup tool geometry."""
    base_dims = jp.array([0.05, 0.05, 0.12])
    cup_radius = 0.03
    cup_height = 0.04
    
    collision_geoms = [
        {
            'type': 'cylinder',
            'pos': [0, 0, 0.06],  # Base cylinder
            'euler': [0, 0, 0],
            'size': [0.025, 0.025, 0.06],
            'name': 'suction_base'
        },
        {
            'type': 'sphere',
            'pos': [0, 0, 0.12 + 0.02],  # Cup (simplified as sphere)
            'euler': [0, 0, 0],
            'size': [cup_radius, cup_radius, cup_height/2],
            'name': 'suction_cup'
        }
    ]
    
    visual_geoms = collision_geoms.copy()
    
    return ToolGeometry(
        tool_type=TOOL_SUCTION,
        mass=0.3,
        dims=base_dims,
        com_offset=jp.array([0, 0, 0.04]),
        collision_geoms=collision_geoms,
        visual_geoms=visual_geoms,
        contact_aabb_pos=jp.array([0, 0, -0.02]),
        contact_aabb_size=jp.array([0.05, 0.05, 0.01])
    )


def create_screwdriver_geometry() -> ToolGeometry:
    """Create the screwdriver tool geometry."""
    handle_radius = 0.015
    handle_length = 0.12
    shaft_radius = 0.005
    shaft_length = 0.15
    bit_length = 0.04
    
    collision_geoms = [
        {
            'type': 'cylinder',  # Handle
            'pos': [0, 0, handle_length/2],
            'euler': [0, 0, 0],
            'size': [handle_radius, handle_radius, handle_length/2],
            'name': 'screwdriver_handle'
        },
        {
            'type': 'cylinder',  # Shaft
            'pos': [0, 0, handle_length + shaft_length/2],
            'euler': [0, 0, 0],
            'size': [shaft_radius, shaft_radius, shaft_length/2],
            'name': 'screwdriver_shaft'
        },
        {
            'type': 'box',  # Bit (simplified as box)
            'pos': [0, 0, handle_length + shaft_length + bit_length/2],
            'euler': [0, 0, 0],
            'size': [0.003, 0.003, bit_length/2],
            'name': 'screwdriver_bit'
        }
    ]
    
    visual_geoms = collision_geoms.copy()
    
    return ToolGeometry(
        tool_type=TOOL_SCREWDRIVER,
        mass=0.4,
        dims=jp.array([0.03, 0.03, handle_length + shaft_length + bit_length]),
        com_offset=jp.array([0, 0, 0.08]),
        collision_geoms=collision_geoms,
        visual_geoms=visual_geoms,
        contact_aabb_pos=jp.array([0, 0, -0.2]),
        contact_aabb_size=jp.array([0.005, 0.005, 0.01])
    )


def create_wrench_geometry() -> ToolGeometry:
    """Create the socket wrench tool geometry."""
    handle_length = 0.15
    handle_radius = 0.01
    head_size = jp.array([0.06, 0.06, 0.04])
    socket_size = 0.01  # 10mm socket
    
    collision_geoms = [
        {
            'type': 'cylinder',  # Handle
            'pos': [0, 0, handle_length/2],
            'euler': [0, 0, 0],
            'size': [handle_radius, handle_radius, handle_length/2],
            'name': 'wrench_handle'
        },
        {
            'type': 'box',  # Head
            'pos': [0, 0.03, handle_length + head_size[2]/2],
            'euler': [0, 0, 0],
            'size': head_size / 2,
            'name': 'wrench_head'
        },
        {
            'type': 'cylinder',  # Socket (simplified as cylinder)
            'pos': [0, 0.03, handle_length + head_size[2] + 0.01],
            'euler': [0, 0, 0],
            'size': [socket_size/2, socket_size/2, 0.01],
            'name': 'wrench_socket'
        }
    ]
    
    visual_geoms = collision_geoms.copy()
    
    return ToolGeometry(
        tool_type=TOOL_WRENCH,
        mass=0.7,
        dims=jp.array([0.06, 0.12, handle_length + head_size[2] + 0.02]),
        com_offset=jp.array([0, 0, 0.05]),
        collision_geoms=collision_geoms,
        visual_geoms=visual_geoms,
        contact_aabb_pos=jp.array([0, 0, -0.03]),
        contact_aabb_size=jp.array([0.02, 0.02, 0.02])
    )


def create_tool_geometry(tool_type: int) -> ToolGeometry:
    """Create the geometry for the specified tool type."""
    if tool_type == TOOL_GRIPPER:
        return create_gripper_geometry()
    elif tool_type == TOOL_SUCTION:
        return create_suction_geometry()
    elif tool_type == TOOL_SCREWDRIVER:
        return create_screwdriver_geometry()
    elif tool_type == TOOL_WRENCH:
        return create_wrench_geometry()
    else:
        raise ValueError(f"Unknown tool type: {tool_type}")
