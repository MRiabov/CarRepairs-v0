# Tooling System for Repairs Environment

This module implements a tooling system for a robotic arm in the Repairs environment, supporting multiple end effectors and a tool changing mechanism.

## Features

- **Multiple Tool Types**:
  - Parallel Jaw Gripper
  - Suction Cup
  - Screwdriver (Phillips/Flat)
  - Socket Wrench

- **Tool Stand**:
  - 6-slot tool stand with magnetic retention
  - Visual and collision geometry for each tool
  - Tool change detection and handling

- **Tool State Management**:
  - Track currently equipped tool
  - Handle tool activation/deactivation
  - Manage tool-object interactions

## Usage

### Basic Setup

```python
from repairs import ToolStand, ToolState, update_tool_state
import jax.numpy as jp

# Create a tool stand with default tools
tool_stand = ToolStand()

# Create initial tool state (no tool equipped)
tool_state = ToolState()

# Simulate picking up a tool
action = {'tool_change_slot': 0}  # Pick up tool from slot 0
ee_pos = tool_stand.slots[0].pos  # End effector at tool position
ee_rot = jp.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation

tool_state, info = update_tool_state(
    tool_state, ee_pos, ee_rot, action, tool_stand
)
```

### Running the Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python examples/tooling_demo.py
```

## API Reference

### `ToolStand`

Manages the tool stand and tool changes.

- `__init__(num_slots=6)`: Initialize with specified number of slots.
- `find_available_slot(tool_type=TOOL_NONE)`: Find an available slot.
- `get_tool_change_pose(slot_idx)`: Get the desired end effector pose for tool change.
- `can_change_tool(ee_pos, ee_rot, slot_idx)`: Check if tool change is possible.
- `change_tool(current_tool, slot_idx)`: Perform a tool change.

### `ToolState`

Represents the state of the currently equipped tool.

- `tool_type`: Type of the currently equipped tool.
- `geometry`: Geometry of the current tool.
- `is_active`: Whether the tool is currently active.
- `activate(target_pos=None, target_rot=None)`: Activate the tool.
- `deactivate()`: Deactivate the tool.

### `update_tool_state(tool_state, ee_pos, ee_rot, action, tool_stand, dt=0.02)`

Update the tool state based on the current action.

## Visualization

The `visualization` module provides functions to visualize the tool stand and tools in 3D using Matplotlib.

Example:
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from repairs import visualization

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
visualization.plot_tool_stand(ax, tool_stand, tool_state)
plt.show()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
