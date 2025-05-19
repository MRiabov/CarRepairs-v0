# Tooling Geometry Specifications

This document details the geometric and physical properties of all tools and the tool stand in the simulation. 

> **Note on End Effectors**:
> - Prefer using end effectors from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) collection when possible.
> - These models are high-quality, physically accurate, and well-tested.
> - They include proper collision meshes and realistic physical properties.
> - Using these models ensures consistency with other MuJoCo-based simulations.
> - Custom tools should only be created when no suitable Menagerie model exists.

## Implementation Guidelines

1. **Model Sources**:
   - First check [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for existing models
   - Use the `mjx`-compatible versions when available
   - Document any modifications made to the original models

2. **Custom Tools**:
   - Only create custom tools when necessary
   - Follow the same conventions as Menagerie models
   - Include proper collision meshes and physical properties
   - Document the design decisions and specifications

## Tool Stand

### Physical Properties
- **Position**: Fixed at [x=0.8, y=0.0, z=0.9] meters in world coordinates
- **Dimensions**: 0.4m (W) × 0.3m (D) × 0.05m (H)
- **Orientation**: Flat on workbench, aligned with world axes
- **Material**: Powder-coated steel with rubberized base
- **Mounting**: Fixed to workbench

### Tool Slots
- **Quantity**: 6 slots arranged in a 2×3 grid
- **Slot Dimensions**: 0.12m × 0.12m
- **Slot Spacing**: 0.15m center-to-center
- **Retention**: Magnetic mounts with alignment guides

## Tools

### 1. Parallel Jaw Gripper
- **Base Dimensions**: 0.1m (W) × 0.1m (D) × 0.15m (H)
- **Jaw Length**: 0.08m
- **Max Opening**: 0.1m
- **Mass**: 0.5kg
- **Center of Mass**: 0.05m from base
- **Collision**: Box collider for base, two box colliders for jaws
- **Contact AABB**: 
  - Position: [0, 0, -0.05] (relative to tool tip)
  - Dimensions: [0.1, 0.1, 0.1]m (covers gripper jaws)
  - Active only when jaws are closed
- **Visual**: Metallic finish with black rubberized grips

### 2. Suction Cup
- **Base Diameter**: 0.05m
- **Height**: 0.12m (extended), 0.08m (compressed)
- **Cup Diameter**: 0.06m
- **Mass**: 0.3kg
- **Center of Mass**: 0.04m from base
- **Collision**: Cylinder collider for base, convex mesh for cup
- **Contact AABB**:
  - Position: [0, 0, -0.02] (just inside cup rim)
  - Dimensions: [0.05, 0.05, 0.01]m (thin volume at cup opening)
  - Active when suction is engaged
- **Visual**: Black rubber cup with metallic base

### 3. Screwdriver (Phillips/Flat)
- **Handle Dimensions**: 0.03m diameter × 0.12m length
- **Shaft Length**: 0.15m
- **Bit Length**: 0.04m
- **Mass**: 0.4kg
- **Center of Mass**: 0.08m from base of handle
- **Collision**: Cylinder collider for handle and shaft
- **Contact AABB**:
  - Position: [0, 0, -0.2] (at bit tip)
  - Dimensions: [0.005, 0.005, 0.01]m (small volume at bit tip)
  - Active during screw driving operations
- **Visual**: Red handle with chrome shaft and bit
- **Bit Types**:
  - PH2 (Phillips #2)
  - PZ2 (Pozidriv #2)
  - SL6 (6mm flat)

### 4. Socket Wrench
- **Handle Length**: 0.15m
- **Head Dimensions**: 0.06m × 0.06m × 0.04m
- **Socket Size**: 0.01m (10mm)
- **Mass**: 0.7kg
- **Center of Mass**: 0.05m from head
- **Collision**: Box collider for handle, convex mesh for head
- **Contact AABB**:
  - Position: [0, 0, -0.03] (at socket opening)
  - Dimensions: [0.02, 0.02, 0.02]m (volume around socket)
  - Active when applying torque
- **Visual**: Black oxide finish with size markings
- **Drive Size**: 1/4" (6.35mm)

## Interaction Volumes

### Contact AABB Definition
Each tool defines an Axis-Aligned Bounding Box (AABB) that represents its active interaction volume. This volume is used to:
- Detect valid tool-object interactions
- Precisely align tools with targets
- Determine successful task completion

### AABB Properties
- **Position**: Relative to the tool's coordinate frame
- **Dimensions**: Size of the interaction volume
- **Activation**: Conditions under which the AABB is active
- **Type**: Type of interaction (grasp, turn, push, etc.)

## Mounting Interface

### Robot Arm Interface
- **Type**: Magnetic quick-release
- **Position**: End of robot arm's 6th axis
- **Orientation**: Z-axis forward, Y-axis up
- **Connection Points**: 4 electromagnetic connectors
- **Alignment**: Tapered guide pins for precise alignment

### Tool Interface
- **Type**: Magnetic mounting plate
- **Dimensions**: 0.08m × 0.08m
- **Connection Points**: 4 ferromagnetic pads
- **Alignment**: Recessed guide holes

## Collision Layers
1. **Default**: Environment and static objects
2. **Robot**: Robot arm and mounted tool
3. **Tools**: Unmounted tools
4. **Interactable**: Objects that can be manipulated
5. **NoCollision**: Visual-only elements

## Physical Properties
- **Friction Coefficients**:
  - Metal-on-metal: 0.5
  - Rubber-on-metal: 0.8
  - Plastic-on-metal: 0.4
- **Restitution**: 0.1 (low bounciness)
- **Damping**: Linear=0.1, Angular=0.1

## Tool Change Parameters
- **Max Approach Distance**: 0.05m
- **Max Angular Error**: 15 degrees
- **Connection Force**: 50N (magnetic)
- **Connection Time**: 0.5s (simulated)
