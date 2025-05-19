"""Visualization utilities for the Repairs environment.

Note: This module uses MuJoCo's MJX for simulation and rendering. MJX is a JAX-based
implementation of MuJoCo that runs efficiently on GPUs/TPUs. For visualization,
we use MuJoCo's native renderer by converting between MJX and MJ data structures.
"""

# Standard library imports
import os
import sys
import tempfile
import webbrowser
from dataclasses import dataclass
from typing import List, Optional

# Third-party imports
import jax
import jax.numpy as jp
import mujoco.mjx as mjx
import numpy as np
from PIL import Image
import mujoco


# Local imports
from repairs.geometry.config import TOOL_NONE, TOOL_STAND_POS, TOOL_SLOT_SPACING

# # Check MuJoCo and MJX versions
# print(f"Python version: {sys.version}")
# # print(f"MuJoCo version: {mujoco.__version__}")
# print(f"MJX version: {mjx.__version__ if hasattr(mjx, '__version__') else 'unknown'}")
# print(f"NumPy version: {np.__version__}")
# print(f"JAX version: {jax.__version__}")

# # Check for version compatibility
# if not hasattr(mujoco, 'MjModel') or not hasattr(mujoco, 'MjData'):
#     print("ERROR: Incompatible MuJoCo version. Make sure you're using MuJoCo 2.3.0 or later.")
#     sys.exit(1)


# Simple system wrapper for MJX
@dataclass
class System:
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData
    mjx_data: mjx.Data
    renderer: Optional[mujoco.Renderer] = None


def create_tool_stand_system(tool_stand) -> System:
    """Create a Brax system containing the tool stand and tools using MJCF.
    
    Args:
        tool_stand: The ToolStand instance to visualize.
        
    Returns:
        A Brax System containing the tool stand and tools.
    """
    # Create an MJCF XML string for the tool stand with slots placeholder
    mjcf_template = """<mujoco>
        <option timestep="0.02" gravity="0 0 -9.81">
            <flag warmstart="enable" gravity="enable" />
        </option>
        <worldbody>
            <body name="tool_stand_base" pos="{pos_x} {pos_y} {pos_z}">
                <geom type="box" size="0.2 0.15 0.05" rgba="0.7 0.7 0.7 1" mass="10.0" />
                {slots}
            </body>
        </worldbody>
    </mujoco>"""
    
    # Generate slot geometries
    slot_geoms = []
    for i, slot in enumerate(tool_stand.slots):
        # Calculate slot position relative to the stand
        # TOOL_SLOT_SPACING is a scalar for both x and y spacing
        x_pos = (i % 3 - 1) * TOOL_SLOT_SPACING
        y_pos = (i // 3) * TOOL_SLOT_SPACING - 0.1
        z_pos = 0.05  # On top of the base
        
        # Add a visual indicator for the slot
        slot_geoms.append(f"""
            <geom name="tool_slot_{i}" 
                  type="cylinder" 
                  pos="{x_pos} {y_pos} {z_pos}" 
                  size="0.02 0.01"
                  euler="90 0 0"
                  rgba="0.9 0.9 0.2 0.7" 
                  contype="0" 
                  conaffinity="0" />
        """)
        
        # If there's a tool in this slot, add its geometry
        if slot.tool_type != TOOL_NONE:
            tool_geom = tool_stand.get_tool_geometry(slot.tool_type)
            # Convert tool geometry to MJCF format (simplified)
            tool_type = "box" if hasattr(tool_geom, 'size') and len(tool_geom.size) == 3 else "cylinder"
            size = " ".join(map(str, tool_geom.size)) if hasattr(tool_geom, 'size') else "0.05 0.05 0.05"
            rgba = " ".join(map(str, tool_geom.rgba)) if hasattr(tool_geom, 'rgba') else "0.5 0.5 0.8 1.0"
            
            slot_geoms.append(f"""
                <geom name="tool_{i}" 
                      type="{tool_type}" 
                      pos="{x_pos} {y_pos} {z_pos + 0.05}" 
                      size="{size}" 
                      rgba="{rgba}" 
                      contype="1" 
                      conaffinity="1" />
            """)
    
    # Format the MJCF string with the slots
    mjcf_str = mjcf_template.format(
        pos_x=float(TOOL_STAND_POS[0]),
        pos_y=float(TOOL_STAND_POS[1]),
        pos_z=float(TOOL_STAND_POS[2]),
        slots="\n".join(slot_geoms)
    )
    
    # Create a temporary file to store the MJCF
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(mjcf_str)
        f.flush()
        temp_path = f.name
    
    # Print the MJCF for debugging
    print("\n=== MJCF Content ===")
    print(mjcf_str)
    print("===================\n")
    
    try:
        # Load the model and create MJX data structures
        print(f"Loading MJCF from: {temp_path}")
        
        # Step 1: Load MJCF model
        print("Creating MuJoCo model...")
        mj_model = mujoco.MjModel.from_xml_path(temp_path)
        print(f"Successfully created MuJoCo model with {mj_model.nq} qpos, {mj_model.nv} qvel")
        
        # Step 2: Create MuJoCo data
        print("Creating MuJoCo data...")
        mj_data = mujoco.MjData(mj_model) 
        # mj_data = mujoco.MjData(mj_model) # note: this is a way to do it.

        print("Successfully created MuJoCo data")
        
        # Step 3: Create MJX model
        print("Creating MJX model...")
        try:
            mjx_model = mjx.put_model(mj_model)
            print("Successfully created MJX model")
        except Exception as e:
            print(f"Error creating MJX model: {e}")
            raise
        
        # Step 4: Create MJX data
        print("Creating MJX data...")
        try:
            mjx_data = mjx.make_data(mjx_model)
            print("Successfully created MJX data")
        except Exception as e:
            print(f"Error creating MJX data: {e}")
            raise
        
        # Step 5: Create System
        print("Creating System...")
        system = System(mj_model, mjx_model, mj_data, mjx_data)
        print("Successfully created System")
        
        return system
    except Exception as e:
        print(f"Error loading MJCF: {e}")
        # Print the first few lines of the file for context
        with open(temp_path, 'r') as f:
            print("\n=== First 20 lines of MJCF file ===")
            for i, line in enumerate(f):
                if i >= 20:
                    break
                print(f"{i+1}: {line.rstrip()}")
            print("==================================")
        raise
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except OSError:
            # Ignore errors if the file was already deleted
            pass


def render_system(
    sys: System,
    duration: float = 2.0,
    framerate: int = 30,
    camera: Optional[str] = None,
) -> List[np.ndarray]:
    """Render frames of the system simulation.
    
    Args:
        sys: The system to render.
        duration: Duration of the simulation in seconds.
        framerate: Frames per second to render.
        camera: Optional camera name to use for rendering.
        
    Returns:
        A list of RGB frames as numpy arrays.
    """
    frames = []
    
    try:
        # Initialize renderer with explicit dimensions
        renderer = mujoco.Renderer(sys.mj_model, height=480, width=640)
        
        # Enable joint visualization
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        
        # Create a copy of the data to avoid modifying the original
        mj_data = mujoco.MjData(sys.mj_model)
        mujoco.mj_resetData(sys.mj_model, mj_data)
        
        # Create MJX data structures
        mjx_model = mjx.put_model(sys.mj_model)
        mjx_data = mjx.put_data(sys.mj_model, mj_data)
        
        # Create a jitted step function
        @jax.jit
        def step(_, mjx_data):
            mjx_data = mjx.step(mjx_model, mjx_data)
            return mjx_data
        
        # Calculate time step and number of steps
        dt = sys.mj_model.opt.timestep
        num_steps = int(duration / dt)
        steps_per_frame = max(1, int(1.0 / (dt * framerate)))
        
        # Run simulation and capture frames
        for step_idx in range(num_steps):
            # Step the simulation
            mjx_data = step(None, mjx_data)
            
            # Render at the specified framerate
            if step_idx % steps_per_frame == 0:
                # Convert MJX data back to MJ data for rendering
                mj_data = mjx.get_data(sys.mj_model, mjx_data)
                
                try:
                    # Update the renderer's scene
                    if camera is not None:
                        renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
                    else:
                        renderer.update_scene(mj_data, scene_option=scene_option)
                    
                    # Render the frame
                    frame = renderer.render()
                    frames.append(frame)
                except Exception as e:
                    print(f"Error rendering frame: {e}")
                    continue
        
        return frames
    except Exception as e:
        print(f"Error in render_system: {e}")
        import traceback
        traceback.print_exc()
        return []


def visualize_tool_stand(
    tool_stand,
    output_file: Optional[str] = None,
    open_browser: bool = True,
    duration: float = 2.0,
    framerate: int = 30,
    camera: Optional[str] = None,
) -> str:
    """Visualize a tool stand with its tools.
    
    Args:
        tool_stand: The tool stand to visualize.
        output_file: Optional path to save the visualization as an HTML file.
        open_browser: Whether to open the visualization in a web browser.
        duration: Duration of the simulation in seconds.
        framerate: Frames per second for the visualization.
        camera: Optional camera name to use for rendering.
        
    Returns:
        Path to the generated HTML file.
    """
    try:
        # Create a system from the tool stand
        sys = create_tool_stand_system(tool_stand)
        
        # Render the system print("Rendering tool stand visualization...")
        frames = render_system(sys, duration=duration, framerate=framerate, camera=camera)
        
        if not frames:
            print("Warning: No frames were rendered.")
            return ""
        
        # Save the first frame as an image
        if output_file is None:
            output_file = "tool_stand_visualization.png"
        
        try:
            img = Image.fromarray(frames[0])
            img.save(output_file)
            print(f"Saved visualization to {output_file}")
            
            # Try to open the image if requested
            if open_browser:
                try:
                    webbrowser.open(f"file://{os.path.abspath(output_file)}")
                except Exception as e:
                    print(f"Could not open in browser: {e}")
            
            return output_file
            
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return ""
            
    except Exception as e:
        print(f"Error in visualize_tool_stand: {e}")
        return ""


def visualize_tool(tool_type: int, output_file: Optional[str] = None, open_browser: bool = True):
    """Visualize a single tool using Brax's HTML renderer.
    
    Args:
        tool_type: The type of tool to visualize (TOOL_* constant).
        output_file: Optional path to save the HTML file. If None, uses a temporary file.
        open_browser: Whether to open the visualization in the default web browser.
        
    Returns:
        The path to the generated HTML file.
    """
    from .tool_stand import ToolStand
    
    # Create a tool stand with one slot
    tool_stand = ToolStand(num_slots=1)
    
    # Replace the first slot with the requested tool
    tool_stand.slots[0].tool_type = tool_type
    tool_stand.slots[0].occupied = True
    
    # Position the tool in the center (using a small offset from the stand)
    tool_stand.slots[0].pos = jp.array([0, 0, 0.15])  # Fixed height above stand
    
    # Visualize
    return visualize_tool_stand(
        tool_stand,
        output_file=output_file,
        open_browser=open_browser
    )


def visualize_tool_change(tool_stand, tool_state, output_file: Optional[str] = None, open_browser: bool = True):
    """Visualize the tool stand with the currently equipped tool highlighted.
    
    Args:
        tool_stand: The ToolStand instance to visualize.
        tool_state: The current tool state.
        output_file: Optional path to save the HTML file. If None, uses a temporary file.
        open_browser: Whether to open the visualization in the default web browser.
        
    Returns:
        The path to the generated HTML file.
    """
    # Create a copy of the tool stand to avoid modifying the original
    from copy import deepcopy
    viz_stand = deepcopy(tool_stand)
    
    # Highlight the currently equipped tool
    if hasattr(tool_state, 'equipped_tool') and tool_state.equipped_tool is not None:
        for i, slot in enumerate(viz_stand.slots):
            if slot.tool_type == tool_state.equipped_tool:
                # The highlighting will be handled in the MJCF generation
                pass
    
    return visualize_tool_stand(viz_stand, output_file=output_file, open_browser=open_browser)
