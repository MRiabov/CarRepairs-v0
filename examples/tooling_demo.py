"""Demo of the tooling system for the Repairs environment using Brax rendering."""

import jax
import jax.numpy as jp
import time

from repairs.geometry.config import (
    TOOL_NONE,
    TOOL_GRIPPER,
    TOOL_SUCTION,
    TOOL_SCREWDRIVER,
    TOOL_WRENCH,
    TOOL_NAMES,
)
from repairs.tool_stand import ToolStand
from repairs.tool_state import ToolState, update_tool_state
from repairs import visualization


def main():
    # Create a tool stand with default tools
    tool_stand = ToolStand()

    # Create an initial tool state (no tool equipped)
    tool_state = ToolState()

    print("=== Tooling System Demo ===")
    print("1. Visualizing the tool stand...")

    # Visualize the tool stand
    html_file = visualization.visualize_tool_stand(tool_stand)
    print(f"Tool stand visualization saved to: {html_file}")

    # Small delay to let the browser open
    time.sleep(2)

    # Visualize each tool individually
    print("\n2. Visualizing individual tools...")
    tools_to_visualize = [
        ("Parallel Jaw Gripper", TOOL_GRIPPER),
        ("Suction Cup", TOOL_SUCTION),
        ("Screwdriver", TOOL_SCREWDRIVER),
        ("Socket Wrench", TOOL_WRENCH),
    ]

    for tool_name, tool_type in tools_to_visualize:
        print(f"  - Visualizing {tool_name}...")
        html_file = visualization.visualize_tool(tool_type)
        time.sleep(1)  # Small delay between tool visualizations

    # Example 3: Simulate tool changes
    print("\n3. Simulating tool changes (see console output)...")
    print("\n=== Tool Change Demo ===")

    # Initial state
    print(f"Initial tool: {TOOL_NAMES.get(tool_state.tool_type, 'None')}")

    # Try to pick up the gripper (slot 0)
    print("\nAttempting to pick up gripper...")
    action = {"tool_change_slot": 0}
    ee_pos = tool_stand.slots[0].pos  # Perfectly aligned
    ee_rot = jp.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation

    tool_state, info = update_tool_state(tool_state, ee_pos, ee_rot, action, tool_stand)
    print(f"Tool change result: {info.get('tool_change', 'Failed')}")
    print(f"Current tool: {TOOL_NAMES.get(tool_state.tool_type, 'None')}")

    # Visualize with the gripper equipped
    print("\nVisualizing with gripper equipped...")
    html_file = visualization.visualize_tool_change(tool_stand, tool_state)
    time.sleep(2)

    # Try to pick up another tool (should fail - already holding a tool)
    print("\nAttempting to pick up suction cup (should fail)...")
    action = {"tool_change_slot": 1}
    ee_pos = tool_stand.slots[1].pos

    tool_state, info = update_tool_state(tool_state, ee_pos, ee_rot, action, tool_stand)
    print(f"Tool change result: {info.get('tool_change', 'Failed')}")

    # Return the gripper to the stand
    print("\nReturning gripper to stand...")
    action = {"tool_change_slot": 0}  # Return to slot 0
    tool_state, info = update_tool_state(tool_state, ee_pos, ee_rot, action, tool_stand)
    print(f"Tool change result: {info.get('tool_change', 'Failed')}")
    print(f"Current tool: {TOOL_NAMES.get(tool_state.tool_type, 'None')}")

    # Visualize with no tool equipped
    print("\nVisualizing with no tool equipped...")
    html_file = visualization.visualize_tool_stand(tool_stand)
    time.sleep(2)

    # Now pick up the suction cup
    print("\nPicking up suction cup...")
    action = {"tool_change_slot": 1}
    ee_pos = tool_stand.slots[1].pos

    tool_state, info = update_tool_state(tool_state, ee_pos, ee_rot, action, tool_stand)
    print(f"Tool change result: {info.get('tool_change', 'Failed')}")
    print(f"Current tool: {TOOL_NAMES.get(tool_state.tool_type, 'None')}")

    # Visualize with the suction cup equipped
    print("\nVisualizing with suction cup equipped...")
    html_file = visualization.visualize_tool_change(tool_stand, tool_state)

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
