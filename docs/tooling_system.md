# Tooling System

This document describes the tooling system for the robotic arm, including available end effectors and the tool changing mechanism. For detailed geometric and physical specifications, see [Tooling Geometry](./tooling_geometry.md).

## Tool State Representation

Each tool has a state represented by an integer:
- `-1`: No tool mounted
- `0`: Parallel Jaw Gripper
- `1`: Suction Cup
- `2`: Screwdriver (Phillips/Flat)
- `3`: Socket Wrench

## End Effector Types

### 1. Parallel Jaw Gripper
- **Purpose**: General purpose grasping
- **Specifications**:
  - Max opening: 10cm
  - Grip force: 20N
  - Weight: 0.5kg
- **Best for**: Holding medium-sized components, battery terminals

### 2. Suction Cup
- **Purpose**: Handling flat, non-porous surfaces
- **Specifications**:
  - Diameter: 5cm
  - Vacuum force: 15N
  - Weight: 0.3kg
- **Best for**: Windshields, body panels, flat components

### 3. Screwdriver (Phillips/Flat)
- **Purpose**: Driving screws and bolts
- **Specifications**:
  - Bit types: PH2, PZ2, Flat 6mm
  - Torque: 5Nm
  - Weight: 0.4kg
- **Best for**: Removing/installing screws, battery terminals

### 4. Socket Wrench
- **Purpose**: Turning nuts and bolts
- **Specifications**:
  - Socket sizes: 8mm - 19mm
  - Torque: 50Nm
  - Weight: 0.7kg
- **Best for**: Wheel nuts, large bolts

## Tool Stand

### Physical Layout
- **Location**: Fixed position within robot's reachable workspace
- **Capacity**: 6 tool slots
- **Orientation**: Vertical mounting for easy access
- **Dimensions**: 30cm x 20cm x 15cm

## Tool Interaction System

### Contact Points and AABBs
Each tool defines precise interaction volumes (AABBs) that determine where and how it can interact with objects:

1. **Gripper**: 
   - Contact AABB at jaw tips
   - Active only when jaws are closed on an object
   - Used for grasping and manipulating components

2. **Screwdriver**:
   - Precise AABB at bit tip
   - Active during screw driving operations
   - Requires exact alignment with screw heads

3. **Socket Wrench**:
   - AABB around socket opening
   - Active when engaging with nuts/bolts
   - Must align with fastener's head

4. **Suction Cup**:
   - AABB at cup opening
   - Active when suction is engaged
   - Requires flush contact with surface

### Tool Change Mechanism

#### Prerequisites for Tool Change
1. Robot must be at the designated tool stand position
2. Arm must be stationary (velocity < 0.01 m/s)
3. No active tool interactions (gripping, screwing, etc.)

#### Tool Change Process
1. **Approach**: Move end effector to tool stand alignment position
2. **Engage**: Activate tool release mechanism
3. **Detect**: System verifies tool release and stand engagement
4. **Select**: New tool is engaged
5. **Verify**: System confirms successful tool attachment

#### Error Conditions
- **Misalignment**: End effector not properly positioned
- **Interference**: Object blocking tool stand
- **Tool Jam**: Mechanical failure in release/attach mechanism
- **Timeout**: Tool change takes too long

#### Recovery Actions
- Retry approach with finer positioning
- Clear any obstructions
- Reset tool stand if jammed
- Emergency stop if multiple failures occur

## Tool Properties and Capabilities

### Physical Properties
- **Mass and Inertia**: Each tool has realistic mass distribution
- **Collision Geometry**: Precise collision meshes for interaction
- **Attachment Points**: Defined connection points for tool mounting
- **Contact AABB**: Precise interaction volume definition

### Functional Properties

#### Gripper
- **Max Force**: 20N clamping force
- **Speed**: 0.1m/s jaw movement
- **Precision**: ±1mm positioning accuracy
- **AABB Activation**: Only when jaws are closed

#### Screwdriver
- **Torque**: 5Nm maximum
- **Bit Types**: PH2, PZ2, SL6
- **RPM**: 0-300 adjustable
- **AABB Activation**: During screw driving motion

#### Socket Wrench
- **Torque**: 50Nm maximum
- **Drive Size**: 1/4" (6.35mm)
- **Socket Sizes**: 8mm - 19mm
- **AABB Activation**: When applying torque

#### Suction Cup
- **Vacuum Strength**: 15N holding force
- **Surface Requirements**: Non-porous, smooth
- **AABB Activation**: When vacuum is engaged
- **Power Consumption**: 5W when active

## Tool State Management

### Tool Status
- **Mounted**: Boolean indicating if tool is currently on the robot
- **Active State**: Tool-specific state (e.g., gripper open/closed, suction on/off)
- **Tool ID**: Unique identifier for the tool type
- **Slot Position**: Position in tool stand (if not mounted)

## Example Tool Change Sequence

1. **Initial State**: `current_tool = -1` (no tool)
2. **Move to Stand**: Position arm at tool stand
3. **Select Tool**: `env.change_tool(tool_id=0)`  # Gripper
4. **Verify**: Check `env.get_current_tool() == 0`
5. **Proceed**: Use gripper for task
6. **Return Tool**: Move to stand and call `env.change_tool(tool_id=-1)`

## Visual Feedback
- **Tool Stand**: Highlights available tools
- **Active Tool**: Visually indicates currently equipped tool
- **Tool Status**: Color-coded indicators for tool condition
- **Tool Tips**: Hover information about each tool's capabilities
