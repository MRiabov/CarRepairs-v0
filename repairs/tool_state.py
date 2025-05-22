"""Tool state and interaction handling for the Repairs environment."""

from typing import Dict, Optional, Tuple, Any
import dataclasses
import jax
import jax.numpy as jp


from repairs.geometry.config import (
    TOOL_NONE,
    TOOL_GRIPPER,
    TOOL_SUCTION,
    TOOL_SCREWDRIVER,
    TOOL_WRENCH,
    TOOL_NAMES,
    Array,
)
from brax import base
from repairs.tools import ToolGeometry


@dataclasses.dataclass
class ToolState:
    """Represents the state of the currently equipped tool."""

    tool_type: int = TOOL_NONE
    geometry: Optional[ToolGeometry] = None
    is_active: bool = False  # Whether the tool is activated (gripping, suction, etc.)
    target_pos: Optional[Array] = None  # Target position for tool actions
    target_rot: Optional[Array] = None  # Target rotation for tool actions

    def get_contact_aabb(self, xform: base.Transform) -> Optional[Tuple[Array, Array]]:
        """Get the contact AABB for the current tool, if any."""
        if self.geometry is None or not self.is_active:
            return None
        return self.geometry.get_contact_aabb(xform)

    def activate(
        self, target_pos: Optional[Array] = None, target_rot: Optional[Array] = None
    ) -> "ToolState":
        """Activate the tool with optional target position/rotation."""
        return dataclasses.replace(
            self,
            is_active=True,
            target_pos=target_pos if target_pos is not None else self.target_pos,
            target_rot=target_rot if target_rot is not None else self.target_rot,
        )

    def deactivate(self) -> "ToolState":
        """Deactivate the tool."""
        return dataclasses.replace(self, is_active=False)


def update_tool_state(
    tool_state: ToolState,
    ee_pos: Array,
    ee_rot: Array,
    action: Dict[str, Any],
    tool_stand: Any,  # Avoid circular import
    dt: float = 0.02,
) -> Tuple[ToolState, Dict[str, Any]]:
    """Update the tool state based on the current action.

    Args:
        tool_state: Current tool state.
        ee_pos: Current end effector position.
        ee_rot: Current end effector rotation (quaternion).
        action: The action dictionary containing tool commands.
        tool_stand: The tool stand instance for tool changes.
        dt: Time step in seconds.

    Returns:
        Updated tool state and info dictionary.
    """
    info = {}

    # Handle tool change command
    if action.get("tool_change_slot", -1) >= 0:
        slot_idx = action["tool_change_slot"]
        if tool_state.tool_type == TOOL_NONE or tool_stand.can_change_tool(
            ee_pos, ee_rot, slot_idx
        ):
            new_tool, success = tool_stand.change_tool(tool_state.tool_type, slot_idx)
            if success:
                tool_state = ToolState(
                    tool_type=new_tool,
                    geometry=tool_stand.get_tool_geometry(new_tool)
                    if new_tool != TOOL_NONE
                    else None,
                )
                info["tool_change"] = (
                    f"Changed to {TOOL_NAMES.get(new_tool, 'unknown')}"
                )
            else:
                info["tool_change"] = "Tool change failed"

    # Handle tool activation
    if "tool_activate" in action and tool_state.tool_type != TOOL_NONE:
        if action["tool_activate"]:
            tool_state = tool_state.activate(ee_pos, ee_rot)
            info["tool_activated"] = True
        else:
            tool_state = tool_state.deactivate()
            info["tool_activated"] = False

    # Tool-specific updates
    if tool_state.tool_type == TOOL_GRIPPER and tool_state.is_active:
        # Gripper-specific logic (e.g., check if grasping an object)
        pass
    elif tool_state.tool_type == TOOL_SUCTION and tool_state.is_active:
        # Suction-specific logic
        pass
    elif tool_state.tool_type == TOOL_SCREWDRIVER and tool_state.is_active:
        # Screwdriver-specific logic
        pass
    elif tool_state.tool_type == TOOL_WRENCH and tool_state.is_active:
        # Wrench-specific logic
        pass

    return tool_state, info


def check_tool_interaction(
    tool_state: ToolState,
    ee_pos: Array,
    ee_rot: Array,
    objects: Dict[str, Any],
    dt: float = 0.02,
) -> Dict[str, Any]:
    """Check for interactions between the tool and objects in the scene.

    Args:
        tool_state: Current tool state.
        ee_pos: Current end effector position.
        ee_rot: Current end effector rotation (quaternion).
        objects: Dictionary of objects in the scene.
        dt: Time step in seconds.

    Returns:
        Dictionary of interaction results.
    """
    if tool_state.tool_type == TOOL_NONE or not tool_state.is_active:
        return {}

    # Get the tool's contact AABB
    xform = base.Transform(pos=ee_pos, rot=ee_rot)
    aabb = tool_state.get_contact_aabb(xform)

    if aabb is None:
        return {}

    aabb_min, aabb_max = aabb
    interactions = {}

    # Check for intersections with objects
    for obj_id, obj in objects.items():
        # Simple AABB intersection test
        obj_pos = obj.get("pos", jp.zeros(3))
        obj_size = obj.get("size", jp.ones(3) * 0.1)  # Default size if not specified
        obj_min = obj_pos - obj_size / 2
        obj_max = obj_pos + obj_size / 2

        # Check for overlap
        overlap = jp.all(aabb_max > obj_min) and jp.all(aabb_min < obj_max)

        if overlap:
            interactions[obj_id] = {
                "tool_type": tool_state.tool_type,
                "position": ee_pos,
                "object_pos": obj_pos,
                "timestamp": jp.array(
                    jax.lax.real_of_complex(jp.array(0.0))
                ),  # Placeholder for actual time
            }

    return interactions
