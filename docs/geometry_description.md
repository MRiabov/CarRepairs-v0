# Geometry System Overview

This document describes the overall geometry system, including the spatial relationships between components, coordinate systems, and interaction rules in the repair simulation environment.

## Coordinate Systems

### World Coordinate System
- **Origin**: Center of the work area
- **X-axis**: Front-to-back of the car (positive = front)
- **Y-axis**: Left-to-right (positive = right)
- **Z-axis**: Up-down (positive = up)
- **Units**: Meters (m)

### Component Local Coordinates
Each component has its own local coordinate system:
- **Origin**: Geometric center of the component
- **Orientation**: Aligned with principal axes of the component
- **Transforms**: Local-to-world transformations are managed by the physics engine

## Component Hierarchy

```
World
├── Workbench (Static)
│   ├── Tool Stand
│   │   └── Tools (various)
│   └── Car Body
│       ├── Engine Bay
│       │   ├── Engine Block
│       │   ├── Oil Fill Cap
│       │   ├── Oil Dipstick
│       │   └── Oil Filter
│       └── Battery System
│           ├── Battery Body
│           ├── Positive Terminal
│           └── Negative Terminal
└── Robot Arm
    └── End Effector
```

## Spatial Relationships

### Engine Bay
- Located in the front of the car
- Contains all engine-related components
- Components are positioned relative to the engine block
- Randomization maintains functional relationships (e.g., oil cap stays on top)

### Battery System
- Typically mounted in the engine compartment
- Must be accessible for removal/replacement
- Terminals must be accessible for connection/disconnection

### Tool Interactions

| Tool          | Interaction Targets               | Interaction Type          | Notes                            |
|---------------|----------------------------------|---------------------------|----------------------------------|
| Gripper       | Battery, Oil Filter              | Grasp/Release            | Primary tool for component manipulation |
| Wrench        | Battery Terminals, Oil Filter    | Rotate/Apply Torque      | For loosening/tightening         |
| Screwdriver  | Small fasteners, Terminals       | Rotate/Apply Torque      | For terminal connections         |
| Suction Cup  | Flat surfaces (e.g., battery top) | Adhere/Release           | For lifting flat components      |


## Collision Handling

### Collision Layers
1. **Static Environment**: Non-movable objects (workbench, tool stand)
2. **Dynamic Objects**: Movable components (battery, tools)
3. **Robot**: Robot arm and end effectors
4. **Sensors**: Contact and proximity sensors

### Collision Response
- **Elastic Collisions**: Used for most object interactions
- **Friction**: Material-based coefficients
- **Damping**: Applied to prevent oscillation

## Physics Parameters

### Material Properties

| Material  | Density (kg/m³) | Friction | Bounce |
|-----------|-----------------|----------|--------|
| Metal     | 7850            | 0.4      | 0.2    |
| Rubber    | 1100            | 0.6      | 0.6    |
| Plastic   | 950             | 0.3      | 0.4    |
| Wood      | 700             | 0.4      | 0.3    |

### Joints and Constraints
- **Battery Terminals**: Fixed joints to battery body
- **Oil Filter**: Threaded connection to engine
- **Oil Cap**: Threaded connection to engine
- **Dipstick**: Sliding joint with limits

## Randomization Strategy

### Position Randomization
1. Engine bay is placed first
2. Battery is placed with minimum distance constraint
3. Components within engine bay are placed relative to engine block
4. Multiple attempts are made to find valid positions

### Size Randomization
- Components scale proportionally
- Mass scales with volume
- Functional relationships are maintained (e.g., terminals stay on battery)

## Performance Considerations

### Collision Meshes
- Simplified convex hulls for complex shapes
- Minimum number of vertices for performance
- Dynamic adjustment of collision detail based on distance from camera

### Level of Detail (LOD)
- High detail: When close to camera
- Medium detail: Mid-range
- Low detail: Far from camera or occluded

## Debug Visualization

### Visual Aids
- Bounding boxes
- Collision volumes
- Contact points
- Force vectors
- Coordinate frames

### Debug Controls
- Toggle collision visualization
- Show/hide components
- Pause simulation
- Step frame-by-frame
