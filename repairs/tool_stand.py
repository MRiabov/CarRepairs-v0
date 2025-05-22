"""Tool stand and tool change system for the Repairs environment."""

from typing import List, Optional, Tuple
import dataclasses
import jax.numpy as jp

from .geometry.config import (
    TOOL_NONE,
    TOOL_GRIPPER,
    TOOL_SUCTION,
    TOOL_SCREWDRIVER,
    TOOL_WRENCH,
    TOOL_STAND_POS,
    TOOL_STAND_DIMS,
    TOOL_SLOT_DIMS,
    TOOL_SLOT_SPACING,
    Array,
)
from .tools import create_tool_geometry, ToolGeometry


@dataclasses.dataclass
class ToolSlot:
    """Represents a slot in the tool stand."""

    pos: Array  # World position of the slot
    tool_type: int  # Type of tool in this slot (TOOL_* constant)
    occupied: bool  # Whether the slot is occupied

    def is_available(self) -> bool:
        """Check if this slot is available for tool change."""
        return not self.occupied


class ToolStand:
    """Manages the tool stand and tool changes."""

    def __init__(self, num_slots: int = 6):
        """Initialize the tool stand with the specified number of slots."""
        self.num_slots = num_slots
        self.slots = self._initialize_slots()
        self.tool_geometries = {}  # Cache of tool geometries by type

    def _initialize_slots(self) -> List[ToolSlot]:
        """Initialize the tool slots in a 2x3 grid."""
        slots = []
        rows = 2
        cols = (self.num_slots + rows - 1) // rows  # Ceiling division

        start_x = -TOOL_SLOT_SPACING * (cols - 1) / 2
        start_y = -TOOL_SLOT_SPACING * (rows - 1) / 2

        # Default tool arrangement
        tool_types = [
            TOOL_GRIPPER,
            TOOL_SUCTION,
            TOOL_SCREWDRIVER,
            TOOL_WRENCH,
            TOOL_NONE,
            TOOL_NONE,
        ]

        for i in range(self.num_slots):
            row = i // cols
            col = i % cols

            # Calculate position relative to stand center
            rel_x = start_x + col * TOOL_SLOT_SPACING
            rel_y = start_y + row * TOOL_SLOT_SPACING

            # Convert to world coordinates
            pos = TOOL_STAND_POS + jp.array(
                [rel_x, rel_y, TOOL_STAND_DIMS[2] / 2 + TOOL_SLOT_DIMS[2] / 2]
            )

            tool_type = tool_types[i] if i < len(tool_types) else TOOL_NONE
            slots.append(
                ToolSlot(
                    pos=pos, tool_type=tool_type, occupied=(tool_type != TOOL_NONE)
                )
            )

        return slots

    def get_tool_geometry(self, tool_type: int) -> ToolGeometry:
        """Get the geometry for a tool type, creating it if necessary."""
        if tool_type not in self.tool_geometries:
            self.tool_geometries[tool_type] = create_tool_geometry(tool_type)
        return self.tool_geometries[tool_type]

    def find_available_slot(self, tool_type: int = TOOL_NONE) -> Optional[int]:
        """Find an available slot that matches the requested tool type.

        Args:
            tool_type: The type of tool to look for, or TOOL_NONE for any tool.

        Returns:
            The index of an available slot, or None if no matching slot is available.
        """
        for i, slot in enumerate(self.slots):
            if slot.is_available() and (
                tool_type == TOOL_NONE or slot.tool_type == tool_type
            ):
                return i
        return None

    def get_tool_change_pose(self, slot_idx: int) -> Tuple[Array, Array]:
        """Get the desired end effector pose for changing tools at the specified slot.

        Args:
            slot_idx: Index of the tool slot.

        Returns:
            A tuple of (position, quaternion) for the end effector.
        """
        if slot_idx < 0 or slot_idx >= len(self.slots):
            raise ValueError(f"Invalid slot index: {slot_idx}")

        slot = self.slots[slot_idx]

        # Position: Align tool tip with slot center
        pos = slot.pos

        # Orientation: Pointing down (Z-down) for tool pickup
        # This assumes the tool's Z-axis points away from the robot when mounted
        rot = jp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (aligned with world)

        return pos, rot

    def can_change_tool(self, ee_pos: Array, ee_rot: Array, slot_idx: int) -> bool:
        """Check if the end effector is in position to change tools.

        Args:
            ee_pos: Current end effector position.
            ee_rot: Current end effector rotation (quaternion).
            slot_idx: Index of the target tool slot.

        Returns:
            True if the end effector is in position for tool change.
        """
        if slot_idx < 0 or slot_idx >= len(self.slots):
            return False

        target_pos, target_rot = self.get_tool_change_pose(slot_idx)

        # Check position error
        pos_error = jp.linalg.norm(ee_pos - target_pos)

        # Check orientation error (quaternion distance)
        rot_error = 1.0 - jp.abs(jp.dot(ee_rot, target_rot)) ** 2

        # Allow small errors in position and orientation
        return (pos_error < 0.01) and (rot_error < 0.01)  # 1cm, ~5 degrees

    def change_tool(self, current_tool: int, slot_idx: int) -> Tuple[int, bool]:
        """Perform a tool change operation.

        Args:
            current_tool: The currently equipped tool type.
            slot_idx: The index of the slot to interact with.

        Returns:
            A tuple of (new_tool, success).
        """
        if slot_idx < 0 or slot_idx >= len(self.slots):
            return current_tool, False

        slot = self.slots[slot_idx]

        # If we're picking up a tool
        if current_tool == TOOL_NONE and slot.occupied:
            new_tool = slot.tool_type
            slot.occupied = False
            return new_tool, True

        # If we're dropping off a tool
        elif (
            current_tool != TOOL_NONE
            and not slot.occupied
            and slot.tool_type == TOOL_NONE
        ):
            slot.tool_type = current_tool
            slot.occupied = True
            return TOOL_NONE, True

        # If we're swapping tools (not allowed in this implementation)
        elif current_tool != TOOL_NONE and slot.occupied:
            return current_tool, False

        return current_tool, False
