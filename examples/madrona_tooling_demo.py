"""Demo of the tooling system for the Repairs environment using Madrona rendering."""

import jax
import jax.numpy as jp
import time
import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repairs.geometry.config import (
    TOOL_NONE, TOOL_GRIPPER, TOOL_SUCTION, TOOL_SCREWDRIVER, TOOL_WRENCH, TOOL_NAMES
)
from repairs.tool_stand import ToolStand
from repairs.tool_state import ToolState, update_tool_state

# Import both visualization modules
from repairs import visualization
from repairs.madrona_visualization import MadronaVisualizer, visualize_madrona_env

class ToolingDemo:
    def __init__(self, use_madrona=True):
        """Initialize the tooling demo.
        
        Args:
            use_madrona: Whether to use Madrona for visualization (default: True)
        """
        self.use_madrona = use_madrona
        self.tool_stand = ToolStand()
        self.tool_state = ToolState()
        self.ee_rot = jp.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        
        if self.use_madrona:
            print("Initializing Madrona visualizer...")
            self.visualizer = MadronaVisualizer(env_name="tool_stand")
        
    def visualize_tool_stand(self):
        """Visualize the current tool stand state."""
        if self.use_madrona:
            # Use Madrona visualization
            output_file = f"tool_stand_{int(time.time())}.html"
            return visualize_madrona_env(
                env_name="tool_stand",
                num_steps=10,  # Just a few steps for static visualization
                output_file=output_file,
                open_browser=True
            )
        else:
            # Fall back to default visualization
            return visualization.visualize_tool_stand(self.tool_stand)
    
    def visualize_tool(self, tool_type):
        """Visualize a specific tool."""
        if self.use_madrona:
            # For Madrona, we'll just show the tool stand with the tool highlighted
            print(f"Madrona visualization for tool {TOOL_NAMES.get(tool_type, 'unknown')}...")
            return self.visualize_tool_stand()
        else:
            return visualization.visualize_tool(tool_type)
    
    def visualize_tool_change(self):
        """Visualize the current tool state."""
        if self.use_madrona:
            # For Madrona, we can show the tool in the environment
            return self.visualize_tool_stand()
        else:
            return visualization.visualize_tool_change(self.tool_stand, self.tool_state)
    
    def change_tool(self, slot_idx):
        """Change the currently equipped tool.
        
        Args:
            slot_idx: Index of the tool slot to interact with
        """
        if slot_idx < 0 or slot_idx >= len(self.tool_stand.slots):
            print(f"Invalid slot index: {slot_idx}")
            return False
            
        ee_pos = self.tool_stand.slots[slot_idx].pos
        action = {'tool_change_slot': slot_idx}
        
        self.tool_state, info = update_tool_state(
            self.tool_state, 
            ee_pos, 
            self.ee_rot, 
            action, 
            self.tool_stand
        )
        
        result = info.get('tool_change', 'Failed')
        tool_name = TOOL_NAMES.get(self.tool_state.tool_type, 'None')
        print(f"Tool change result: {result}")
        print(f"Current tool: {tool_name}")
        
        return result == 'success'

def main():
    print("=== Tooling System Demo with Madrona Visualization ===")
    
    # Create demo instance with Madrona visualization
    demo = ToolingDemo(use_madrona=True)
    
    print("1. Visualizing the tool stand...")
    html_file = demo.visualize_tool_stand()
    print(f"Tool stand visualization saved to: {html_file}")
    time.sleep(2)  # Small delay to let the browser open
    
    # Visualize each tool individually
    print("\n2. Visualizing individual tools...")
    tools_to_visualize = [
        ("Parallel Jaw Gripper", TOOL_GRIPPER),
        ("Suction Cup", TOOL_SUCTION),
        ("Screwdriver", TOOL_SCREWDRIVER),
        ("Socket Wrench", TOOL_WRENCH)
    ]
    
    for tool_name, tool_type in tools_to_visualize:
        print(f"  - Visualizing {tool_name}...")
        demo.visualize_tool(tool_type)
        time.sleep(1)  # Small delay between tool visualizations
    
    # Tool change demo
    print("\n3. Simulating tool changes...")
    print("\n=== Tool Change Demo ===")
    
    # Initial state
    print(f"Initial tool: {TOOL_NAMES.get(demo.tool_state.tool_type, 'None')}")
    
    # Pick up the gripper
    print("\nAttempting to pick up gripper...")
    if demo.change_tool(0):  # Gripper is in slot 0
        print("Visualizing with gripper equipped...")
        demo.visualize_tool_change()
        time.sleep(2)
    
    # Try to pick up another tool (should fail - already holding a tool)
    print("\nAttempting to pick up suction cup (should fail)...")
    demo.change_tool(1)  # Suction cup is in slot 1
    
    # Return the gripper to the stand
    print("\nReturning gripper to stand...")
    if demo.change_tool(0):  # Return to slot 0
        print("Visualizing with no tool equipped...")
        demo.visualize_tool_stand()
        time.sleep(2)
    
    # Now pick up the suction cup
    print("\nPicking up suction cup...")
    if demo.change_tool(1):  # Suction cup is in slot 1
        print("Visualizing with suction cup equipped...")
        demo.visualize_tool_change()
    
    print("\n=== Demo Complete ===")
    print("Check the generated HTML files for visualizations.")

if __name__ == "__main__":
    main()
