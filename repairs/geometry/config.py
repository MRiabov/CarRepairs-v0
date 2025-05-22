"""Configuration constants for the Repairs environment."""

from typing import Dict, Any, Tuple
import jax.numpy as jp

# Tool types
TOOL_NONE = -1
TOOL_GRIPPER = 0
TOOL_SUCTION = 1
TOOL_SCREWDRIVER = 2
TOOL_WRENCH = 3

# Tool names for reference
TOOL_NAMES = {
    TOOL_NONE: "No Tool",
    TOOL_GRIPPER: "Parallel Jaw Gripper",
    TOOL_SUCTION: "Suction Cup",
    TOOL_SCREWDRIVER: "Screwdriver",
    TOOL_WRENCH: "Socket Wrench",
}

# Tool stand configuration
TOOL_STAND_POS = jp.array([0.8, 0.0, 0.9])  # World position
TOOL_STAND_DIMS = jp.array([0.4, 0.3, 0.05])  # Width, Depth, Height
TOOL_SLOT_SPACING = 0.15  # Center-to-center distance between slots
TOOL_SLOT_DIMS = jp.array([0.12, 0.12, 0.1])  # Slot dimensions

# Mounting interface configuration
MOUNT_OFFSET = jp.array([0.0, 0.0, 0.1])  # Offset from robot flange to tool base

# Re-export common types
Array = jp.ndarray
PyTree = Any
