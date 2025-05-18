# Visualization System

This document outlines the visualization components and camera setup for the car repair simulation environment.

## Camera System

### Fixed Overhead Cameras

Three fixed cameras provide different angled views of the car hood area, all positioned 2.3m above ground level.

1. **Front-Left View**
   - Position: [x=-1.0, y=-1.0, z=2.3] meters
   - Target: Center of car hood
   - FOV: 60 degrees
   - Purpose: Diagonal view of work area

2. **Top-Down View**
   - Position: [x=0.0, y=0.0, z=2.3] meters (directly above center of hood)
   - Target: Center of car hood
   - FOV: 45 degrees
   - Purpose: Orthographic top-down view

3. **Front-Right View**
   - Position: [x=1.0, y=-1.0, z=2.3] meters
   - Target: Center of car hood
   - FOV: 60 degrees
   - Purpose: Complementary diagonal view

### Camera Controls
- **Pan/Tilt**: Adjust camera angle
- **Zoom**: Adjust field of view
- **Follow Mode**: Track specific components or tools
- **Save/Load Views**: Store and recall camera positions

## Visual Feedback

### Task Guidance
- **Current Objective**: display
- **Next Steps**: An array with descriptions of next steps.

### Interactive Elements
- **Highlighting**:
  - Active tool: Blue glow
  - Target component: Pulsing green outline
  - Incorrect interaction: Red flash
- **Trajectory Preview**: Ghosted path of intended motion
- **Collision Warning**: Visual indicators for potential collisions

### Debug Information
- **Coordinate Axes**: Show world and local coordinate systems
- **Force Vectors**: Visualize applied forces
- **Contact Points**: Highlight collision interactions
- **Joint Angles**: Display current robot configuration

## Rendering Settings
Rendering is currently done only using brax native rendering, for speed. You should not use Matplotlib for rendering, use only brax as it comes with its own rendering system, which is preset for what we need.

## Debug Visualization

### Collision Visualization
- Display contact points

### Performance Metrics
- Frame rate counter
- Memory usage
- Physics step timing
