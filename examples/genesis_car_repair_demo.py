"""
Car Repair Demo using Genesis Simulator

This script demonstrates the car repair environment using the Genesis simulator,
showing the workbench, car components, and tool stand.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import genesis as gs
from jax import random

# Add the repo root to the path so we can import the repairs package
sys.path.append(str(Path(__file__).parent.parent))

from repairs.geometry.genesis_geometry import create_default_genesis_world


def setup_scene() -> gs.Scene:
    """Set up the Genesis scene with appropriate settings."""
    # Initialize Genesis
    gs.init(backend=gs.cpu, logging_level="info")

    # Create a headless scene
    scene = gs.Scene(
        show_viewer=False, renderer=gs.renderers.Rasterizer(), gravity=(0, 0, -9.81)
    )

    return scene


def setup_camera(scene: gs.Scene):
    """Set up the camera for the scene."""
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(3.0, -3.0, 2.5),  # Position the camera to view the scene
        lookat=(1.0, 0.0, 0.5),  # Look at the car
        fov=45,
        GUI=False,
    )
    return camera


def main():
    """Run the car repair demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Car Repair Demo with Genesis")
    parser.add_argument(
        "--output",
        type=str,
        default="car_repair_demo.mp4",
        help="Output video file path",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Simulation duration in seconds"
    )
    args = parser.parse_args()

    # Set up the scene
    scene = setup_scene()

    # Create the world
    _ = create_default_genesis_world(scene=scene, key=random.PRNGKey(42))

    # Set up camera
    camera = setup_camera(scene)

    # Build the scene
    scene.build()

    # Start recording
    print(f"Starting simulation and recording to {args.output}...")
    camera.start_recording()

    # Run the simulation
    fps = 60
    num_steps = int(args.duration * fps)

    for i in range(num_steps):
        # Update camera to follow the action
        if i % 30 == 0:  # Update camera every half second
            camera.set_pose(
                pos=(3.0 + 0.5 * np.sin(i / 60), -3.0 + 0.5 * np.cos(i / 60), 2.5),
                lookat=(1.0, 0.0, 0.5),
            )

        # Step the simulation
        scene.step()

        # Render the current frame
        camera.render()

    # Save the recording
    camera.stop_recording(save_to_filename=args.output, fps=fps)
    print(f"Simulation complete. Video saved to {args.output}")


if __name__ == "__main__":
    main()
