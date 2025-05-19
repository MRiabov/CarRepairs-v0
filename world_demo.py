"""
Car Repairs World Demo using Genesis Simulator

This script demonstrates the Car Repairs environment using the Genesis simulator,
shcowing the workbench, car components, and tool stand.
"""
import os
import sys
import numpy as np
import genesis as gs
from pathlib import Path

# Add the repo root to the path so we can import the repairs package
sys.path.append(str(Path(__file__).parent))

from repairs.geometry.car_components import CarFront, create_car_components
from repairs.geometry.config import (
    TOOL_GRIPPER, TOOL_SUCTION, TOOL_SCREWDRIVER, TOOL_WRENCH,
    TOOL_STAND_POS, TOOL_STAND_DIMS, TOOL_SLOT_SPACING, TOOL_SLOT_DIMS
)

def create_workbench(scene, position=(0, 0, 0), dims=(3.0, 2.0, 0.1)):
    """Create a workbench in the scene."""
    # Main surface
    surface = scene.add_entity(
        gs.morphs.Box(half_extents=np.array(dims)/2),
        position=position,
        material=gs.materials.Rigid(color=(0.6, 0.6, 0.6, 1.0)),
        name="workbench_surface"
    )
    
    # Add legs
    leg_size = np.array([0.05, 0.05, 0.4])
    leg_positions = [
        [dims[0]/2 - 0.1, dims[1]/2 - 0.1, -leg_size[2]/2],
        [dims[0]/2 - 0.1, -dims[1]/2 + 0.1, -leg_size[2]/2],
        [-dims[0]/2 + 0.1, dims[1]/2 - 0.1, -leg_size[2]/2],
        [-dims[0]/2 + 0.1, -dims[1]/2 + 0.1, -leg_size[2]/2]
    ]
    
    for i, pos in enumerate(leg_positions):
        scene.add_entity(
            gs.morphs.Box(half_extents=leg_size/2),
            position=pos,
            material=gs.materials.Rigid(color=(0.4, 0.4, 0.4, 1.0)),
            name=f"workbench_leg_{i}"
        )
    
    return surface

def create_tool_stand(scene, position=TOOL_STAND_POS, dims=TOOL_STAND_DIMS):
    """Create a tool stand with tool slots."""
    # Base of the tool stand
    base = scene.add_entity(
        gs.morphs.Box(half_extents=np.array(dims)/2),
        position=position,
        material=gs.materials.Rigid(color=(0.3, 0.3, 0.3, 1.0)),
        name="tool_stand_base"
    )
    
    # Tool slots (simplified as colored boxes for now)
    tool_colors = {
        TOOL_GRIPPER: (0.2, 0.8, 0.2, 1.0),    # Green
        TOOL_SUCTION: (0.8, 0.2, 0.2, 1.0),    # Red
        TOOL_SCREWDRIVER: (0.2, 0.2, 0.8, 1.0), # Blue
        TOOL_WRENCH: (0.8, 0.8, 0.2, 1.0)      # Yellow
    }
    
    tool_types = [TOOL_GRIPPER, TOOL_SUCTION, TOOL_SCREWDRIVER, TOOL_WRENCH]
    
    for i, tool_type in enumerate(tool_types):
        row = i // 2
        col = i % 2
        tool_pos = [
            position[0] + (col - 0.5) * TOOL_SLOT_SPACING,
            position[1] + (0.5 - row) * TOOL_SLOT_SPACING,
            position[2] + dims[2]/2 + 0.01
        ]
        scene.add_entity(
            gs.morphs.Cylinder(radius=0.05, height=0.1),
            position=tool_pos,
            material=gs.materials.Rigid(color=tool_colors[tool_type]),
            name=f"tool_{tool_type}"
        )
    
    return base

def create_car_front(scene, position=(1.5, 0, 0.1)):
    """Create a simplified car front with engine bay."""
    # Car body (simplified as a box for now)
    car_body = scene.add_entity(
        gs.morphs.Box(half_extents=np.array([1.0, 0.8, 0.4])/2),
        position=position,
        material=gs.materials.Rigid(color=(0.1, 0.1, 0.1, 1.0)),
        name="car_body"
    )
    
    # Engine bay (simplified as a box with a lid)
    engine_bay = scene.add_entity(
        gs.morphs.Box(half_extents=np.array([0.8, 0.6, 0.2])/2),
        position=[position[0], position[1], position[2] + 0.2],
        material=gs.materials.Rigid(color=(0.3, 0.3, 0.3, 1.0)),
        name="engine_bay"
    )
    
    # Engine bay lid (slightly open)
    lid = scene.add_entity(
        gs.morphs.Box(half_extents=np.array([0.8, 0.6, 0.02])/2),
        position=[position[0], position[1], position[2] + 0.4],
        euler_angles=[0.2, 0, 0],  # Slightly open
        material=gs.materials.Rigid(color=(0.4, 0.4, 0.4, 1.0)),
        name="engine_bay_lid"
    )
    
    return car_body

def main():
    """Main function to run the Genesis simulation."""
    # Initialize Genesis
    gs.init(backend=gs.cpu, logging_level='info')
    
    # Create a headless scene
    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        gravity=(0, 0, -9.81)
    )
    
    # Add floor
    floor = scene.add_entity(
        gs.morphs.Plane(size=20.0),
        material=gs.materials.Rigid(color=(0.2, 0.3, 0.2, 1.0)),
        name="floor"
    )
    
    # Create workbench
    workbench = create_workbench(scene, position=(0, 0, 0.2))
    
    # Create tool stand
    tool_stand = create_tool_stand(scene)
    
    # Create car front on workbench
    car = create_car_front(scene)
    
    # Add a camera
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(3.0, -3.0, 2.5),  # Position the camera to view the scene
        lookat=(1.0, 0.0, 0.5),  # Look at the car
        fov=45,
        GUI=False
    )
    
    # Build the scene
    scene.build()
    
    # Start recording
    output_file = "car_repairs_demo.mp4"
    camera.start_recording()
    print(f"Starting simulation and recording to {output_file}...")
    
    # Run the simulation
    for i in range(300):  # 5 seconds at 60 FPS
        scene.step()
        camera.render()
    
    # Save the recording
    camera.stop_recording(save_to_filename=output_file, fps=60)
    print(f"Simulation complete. Video saved to {output_file}")

if __name__ == "__main__":
    main()
