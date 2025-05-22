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


def main():
    """Main function to run the Genesis simulation."""
    print("Starting Genesis simulation...")
    # Initialize Genesis
    gs.init(backend=gs.cpu, logging_level="debug")

    # Create a headless scene
    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )

    # Add floor
    floor = scene.add_entity(
        gs.morphs.Plane(),
    )

    # # # Create tool stand
    # tool_stand = scene.add_entity(
    #     gs.morphs.Mesh(
    #         file="/home/mriabov/Work/Projects/RepairsComponents-v0/geom_exports/tooling_stands/tool_stand_big.gltf",
    #         pos=(0, 0, 0),
    #         fixed=True,
    #     ),
    #     surface=gs.surfaces.Iron(),
    # )
    # car_bay_1 = scene.add_entity(
    #     gs.morphs.Mesh(
    #         file="/home/mriabov/Work/Projects/Repairs-v0/repairs/geometry/cad/exports/engine_bay_1.gltf",
    #         pos=(500, 0, 0),
    #         fixed=True,
    #     ),
    #     surface=gs.surfaces.Aluminium(),
    # )
    # car_battery = scene.add_entity(
    #     gs.morphs.Mesh(
    #         file="/home/mriabov/Work/Projects/Repairs-v0/repairs/geometry/cad/exports/car_battery.gltf",
    #         pos=(0, 0, 0),
    #         fixed=True,
    #     ),
    #     surface=gs.surfaces.Plastic(color=(0.2, 0.2, 0.2, 1.0)),
    # )
    # robot_support = scene.add_entity(
    #     gs.morphs.Box(size=(0.4, 0.4, 0.7), pos=(0.2, 0.2, 0.35))
    # )

    # franka = scene.add_entity(
    #     gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0.7))
    # )

    # # Create car front on workbench
    # car = create_car_front(scene)

    # Add a camera
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(1000, 1000, 1000),  # Position the camera to view the scene
        lookat=(500, 0, 0),  # Look at the car
        fov=45,
        GUI=False,
    )

    # Build the scene
    scene.build()

    # Start recording
    output_file = "examples/car_repairs_demo.mp4"
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
    print("Running Genesis simulation...")
    main()
