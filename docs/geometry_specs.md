# Geometry Specifications

This document outlines the 3D models and their specifications required for the car repair simulation environment.

## Car Components

### 1. Battery System
- **Battery**: Rectangular prism (30cm x 17cm x 19cm)
- **Terminals**: Two cylindrical connectors (positive and negative)
- **Mounting Bracket**: Metal frame securing the battery in place
- **Cables**: Main positive and ground cables with connectors. Currently implemented as just the final connector.

### 2. Engine Bay
- **Engine Block**: Main engine structure with valve cover
- **Oil Fill Cap**: Circular cap (5cm diameter)
- **Dipstick**: Long, thin metal rod with measurement markings
- **Oil Filter**: Cylindrical component (10cm height, 8cm diameter)

### 3. Wheel Assembly (x4)
- **Tire**: Torus shape with tread pattern
- **Rim**: Metallic wheel structure
- **Hub Cap**: Decorative cover (removable)
- **Lug Nuts**: 4-6 per wheel (hexagonal nuts, 2cm across flats)

### 4. Underbody Components
- **Oil Pan**: Shallow rectangular container under engine
- **Drain Plug**: Hexagonal bolt at bottom of oil pan
- **Transmission Pan**: Similar to oil pan but for transmission fluid

## Robot Arm
- **Base**: Fixed to the workbench
- **Joints**: 6-DOF Franka arm - sourced from Mujoco Menagerie.
- **End Effector Mount**: Standard interface for tool attachment

## Work Area
- **Workbench**: Flat surface for parts and tools
- **Tool Stand**: Dedicated area for storing end effectors
- **Parts Tray**: Area for removed components
- **Car**: The car itself (work environment).

## Collision Geometry
- All components will have simplified collision meshes for physics simulation
- Small parts like bolts will use primitive shapes (cylinders, boxes)
- Complex shapes will use convex hull approximations

## Materials and Textures
- Metal components: Metallic finish with appropriate roughness
- Rubber/Plastic: Non-metallic materials with appropriate textures
- Fluids: Transparent materials with appropriate viscosity simulation

## Scale and Units
- All measurements in meters (SI units)
- Mass properties will be realistically approximated
- Friction and other physical properties will be tuned for realistic interaction
