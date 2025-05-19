"""Madrona-based visualization utilities for the Repairs environment.

This module provides GPU-accelerated visualization using the Madrona engine.
It's designed to work alongside the existing MJX-based visualization.
"""

import os
import sys
from typing import List, Optional, Tuple, Dict, Any
import tempfile
import webbrowser
import jax
import jax.numpy as jp
import numpy as np
from PIL import Image

# Import Madrona components
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground._src.dm_control_suite import make_env
from mujoco_playground.wrappers import wrap_for_brax_training

class MadronaVisualizer:
    """A class to handle Madrona-based visualization of the repair environment."""
    
    def __init__(self, env_name: str = "cartpole_balance", num_envs: int = 1):
        """Initialize the Madrona visualizer.
        
        Args:
            env_name: Name of the environment to visualize
            num_envs: Number of parallel environments to simulate
        """
        self.env_name = env_name
        self.num_envs = num_envs
        self.env = None
        self.state = None
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up the Madrona environment with the specified configuration."""
        # Create the base environment
        env = make_env(self.env_name)
        
        # Wrap it for Brax training with vision
        self.env = wrap_for_brax_training(
            env,
            vision=True,
            num_vision_envs=self.num_envs,
            action_repeat=1,
            episode_length=1000,  # Default episode length
        )
        
        # JIT the reset and step functions
        self.jit_reset = jax.jit(self.env.reset)
        self.jit_step = jax.jit(self.env.step)
        
        # Initialize the environment state
        key = jax.random.PRNGKey(0)
        self.state = self.jit_reset(jax.random.split(key, self.num_envs))
    
    def set_camera_position(self, camera_name: str, position: Tuple[float, float, float], 
                          target: Optional[Tuple[float, float, float]] = None):
        """Set the camera position and target.
        
        Args:
            camera_name: Name of the camera to configure
            position: (x, y, z) position of the camera
            target: (x, y, z) target point the camera looks at
        """
        # Implementation depends on Madrona's camera API
        pass
    
    def render(self, camera: Optional[str] = None) -> np.ndarray:
        """Render the current state of the environment.
        
        Args:
            camera: Optional camera name to use for rendering
            
        Returns:
            RGB image as a numpy array
        """
        # Get the current observation which includes the rendered frames
        obs = self.state.obs
        
        # Find the first camera observation
        for key in obs:
            if key.startswith('pixels/'):
                # Convert JAX array to numpy and return the first environment's view
                return np.array(obs[key][0])
        
        raise ValueError("No camera observations found in the environment")
    
    def step(self, actions: np.ndarray) -> np.ndarray:
        """Step the environment and return the rendered frame.
        
        Args:
            actions: Array of actions to take
            
        Returns:
            Rendered frame after the step
        """
        # Convert actions to JAX array if needed
        if not isinstance(actions, jp.ndarray):
            actions = jp.array(actions)
        
        # Take a step
        self.state = self.jit_step(self.state, actions)
        
        # Return the rendered frame
        return self.render()
    
    def render_multiview(self, cameras: Optional[List[str]] = None) -> List[np.ndarray]:
        """Render multiple views of the environment.
        
        Args:
            cameras: List of camera names to render
            
        Returns:
            List of rendered frames as numpy arrays
        """
        if cameras is None:
            # Default to all available cameras
            frames = []
            for key in self.state.obs:
                if key.startswith('pixels/'):
                    frames.append(np.array(self.state.obs[key][0]))
            return frames
        else:
            return [self.render(cam) for cam in cameras]
    
    def create_animation(self, actions: List[np.ndarray], 
                        cameras: Optional[List[str]] = None) -> Dict[str, List[np.ndarray]]:
        """Create an animation from a sequence of actions.
        
        Args:
            actions: List of actions to take
            cameras: List of camera names to render
            
        Returns:
            Dictionary mapping camera names to lists of frames
        """
        if cameras is None:
            # Get all available cameras
            cameras = []
            for key in self.state.obs:
                if key.startswith('pixels/'):
                    cam_name = key.split('/')[-1]
                    if cam_name not in cameras:
                        cameras.append(cam_name)
        
        # Initialize frame buffers
        frames = {cam: [] for cam in cameras}
        
        # Reset the environment
        key = jax.random.PRNGKey(0)
        self.state = self.jit_reset(jax.random.split(key, self.num_envs))
        
        # Take actions and record frames
        for action in actions:
            # Step the environment
            self.step(action)
            
            # Render from all cameras
            for cam in cameras:
                frame = self.render(cam)
                frames[cam].append(frame)
        
        return frames

def visualize_madrona_env(env_name: str = "cartpole_balance", 
                         num_steps: int = 100,
                         output_file: Optional[str] = None,
                         open_browser: bool = True) -> str:
    """Create and visualize a Madrona environment.
    
    Args:
        env_name: Name of the environment to visualize
        num_steps: Number of steps to simulate
        output_file: Path to save the visualization (HTML file)
        open_browser: Whether to open the visualization in a web browser
        
    Returns:
        Path to the generated HTML file
    """
    # Set up a temporary file if none provided
    if output_file is None:
        fd, output_file = tempfile.mkstemp(suffix='.html')
        os.close(fd)
    
    # Create the visualizer
    visualizer = MadronaVisualizer(env_name=env_name)
    
    # Generate random actions
    key = jax.random.PRNGKey(0)
    action_size = visualizer.env.action_size
    actions = jax.random.uniform(
        key, 
        (num_steps, visualizer.num_envs, action_size),
        minval=-1.0,
        maxval=1.0
    )
    
    # Create animation
    frames_dict = visualizer.create_animation(actions)
    
    # For now, just save the first frame as an image
    # In a real implementation, you would create an interactive HTML viewer
    first_cam = next(iter(frames_dict.keys()))
    first_frame = frames_dict[first_cam][0]
    img = Image.fromarray(first_frame)
    img.save(output_file.replace('.html', '.png'))
    
    # Create a simple HTML viewer
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Madrona Visualization</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px;
                text-align: center;
            }}
            .container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 20px;
            }}
            .frame-container {{
                border: 1px solid #ddd;
                padding: 10px;
                margin: 10px 0;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Madrona Environment: {env_name}</h1>
            <div class="frame-container">
                <h2>First Frame</h2>
                <img src="{img_path}" alt="First frame of simulation">
            </div>
            <div>
                <p>This is a placeholder visualization. In a full implementation, this would show an interactive 3D viewer.</p>
                <p>Check the console for additional output.</p>
            </div>
        </div>
    </body>
    </html>
    """.format(
        env_name=env_name,
        img_path=os.path.basename(output_file.replace('.html', '.png'))
    )
    
    # Write the HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    # Open in browser if requested
    if open_browser:
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
    
    return output_file

if __name__ == "__main__":
    # Example usage
    output = visualize_madrona_env()
    print(f"Visualization saved to: {output}")
