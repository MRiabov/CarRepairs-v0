# Car Geometry Specifications

This document outlines the 3D models, their specifications, and randomization parameters for the car components in the repair simulation environment.

## Randomization Parameters

- **Random Scale**: ±20% variation in component dimensions
- **Minimum Component Distance**: 0.1m between components
- **Position Attempts**: Up to 10 attempts to find non-overlapping positions
- **Random Seed**: All randomization is deterministic based on provided JAX random keys

## Car Components

### 1. Battery System

#### Base Specifications
- **Dimensions**: 30cm × 17cm × 19cm (nominal)
- **Mass**: 15kg (nominal)
- **Material**: Hard plastic case with metal terminals

#### Randomized Properties
- **Dimensions**: ±20% of nominal
- **Mass**: Scaled with volume
- **Position**: Randomly placed in engine compartment

#### Components
1. **Main Battery**
   - Shape: Rectangular prism
   - Color: Light gray
   - Collision: Box collider

2. **Terminals**
   - Positive (red) and Negative (black)
   - Shape: Cylindrical
   - Size: 2-3cm radius, 2cm height
   - Position: Top of battery, randomly offset

### 2. Engine Bay

#### Base Specifications
- **Dimensions**: 1.2m × 0.8m × 0.6m (nominal)
- **Material**: Metal frame with various components

#### Randomized Properties
- **Dimensions**: ±20% of nominal
- **Component Positions**: Randomly placed within bounds
- **Component Sizes**: Scaled with engine bay size

#### Components
1. **Engine Block**
   - Shape: Rectangular prism
   - Size: Proportional to engine bay (33% length, 38% width, 50% height)
   - Color: Dark gray
   - Collision: Box collider

2. **Oil Fill Cap**
   - Shape: Cylindrical
   - Size: 2.5-3.5cm radius, 2cm height
   - Position: Random on top of engine
   - Color: Yellow
   - Collision: Cylinder collider

3. **Oil Dipstick**
   - Shape: Thin cylinder
   - Size: 4-6mm radius, 15-25cm length
   - Position: Random on side of engine
   - Color: Light gray
   - Collision: Capsule collider

4. **Oil Filter**
   - Shape: Cylindrical
   - Size: 7-9cm diameter, 8-12cm height
   - Position: Side of engine
   - Color: Light gray with black stripe
   - Collision: Cylinder collider

## Coordinate System

- **Origin**: Center of work area
- **X-axis**: Front-to-back of car (positive = front)
- **Y-axis**: Left-to-right (positive = right)
- **Z-axis**: Up-down (positive = up)

## Physical Properties

- **Gravity**: 9.81 m/s² downward
- **Friction**:
  - Metal-on-metal: μ = 0.4
  - Rubber-on-metal: μ = 0.6
  - Plastic-on-metal: μ = 0.3
- **Bounce**:
  - Metal: 0.2
  - Rubber: 0.6
  - Plastic: 0.4

## Collision Layers

1. **Static Environment**: Non-movable objects (workbench, tool stand)
2. **Dynamic Objects**: Movable components (battery, tools)
3. **Robot**: Robot arm and end effectors
4. **Sensors**: Contact and proximity sensors

## Randomization Ranges

| Component       | Parameter    | Min     | Max     | Unit |
|----------------|--------------|---------|---------|------|
| Battery        | Length       | 0.24    | 0.36    | m    |
|                | Width        | 0.136   | 0.204   | m    |
|                | Height       | 0.152   | 0.228   | m    |
| Engine Bay     | Length       | 0.96    | 1.44    | m    |
|                | Width        | 0.64    | 0.96    | m    |
|                | Height       | 0.48    | 0.72    | m    |
| Oil Fill Cap  | Radius       | 0.025   | 0.035   | m    |
| Oil Dipstick  | Length       | 0.15    | 0.25    | m    |
| Oil Filter    | Diameter     | 0.07    | 0.09    | m    |
|               | Height       | 0.08    | 0.12    | m    |
