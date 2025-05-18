# Environment API

This document describes the programming interface for interacting with the car repair simulation environment.

## Core Interface

### Initialization
```python
# Create and initialize the environment
env = CarRepairEnv(
    task_sequence=default_tasks,  # Optional: Predefined sequence of tasks
    render_mode='human',         # 'human' for visualization, 'rgb_array' for headless
    max_steps=1000,             # Maximum steps per episode
    time_step=0.01              # Physics simulation time step (seconds)
)

# Reset the environment to start a new episode
observation, info = env.reset(task_id=0)  # task_id specifies starting task
```

### Observation Space
```python
observation = {
    # Robot state (13 values)
    'arm_position': np.array([x, y, z]),               # 3D position (meters)
    'arm_orientation': np.array([qx, qy, qz, qw]),    # Orientation (quaternion)
    'arm_velocity': np.array([vx, vy, vz]),            # Linear velocity (m/s)
    'arm_angular_velocity': np.array([wx, wy, wz]),    # Angular velocity (rad/s)
    'gripper_state': float,                            # 0: fully open, 1: fully closed
    'current_tool': int,                               # ID of current tool (-1 for none)
    'tool_state': int,                                 # Tool-specific state
    
    # Task information
    'target_position': np.array([x, y, z]),             # 3D target position
    'target_orientation': np.array([qx, qy, qz, qw]),  # Target orientation (quaternion)
    'target_type': int,                                # Type of target
    'target_id': int,                                  # ID of target object
    'task_complete': bool,                             # Current task status
    'next_steps': List[str],                           # Descriptions of upcoming steps
    
    # Sensor data (camera feeds)
    'images': {
        'front_left': np.uint8([H, W, 3]),   # Front-left diagonal view
        'top_down': np.uint8([H, W, 3]),      # Orthographic top-down view
        'front_right': np.uint8([H, W, 3])    # Front-right diagonal view
    },
    
    # Status and metrics
    'collision': bool,                         # Collision detected
    'timestep': int,                           # Current timestep
    'elapsed_time': float,                     # Elapsed time (seconds)
    'energy_usage': float,                    # Cumulative energy used (Joules)
    'current_reward': float                   # Reward from last action
}
```

### Action Space
```python
action = {
    # Target end-effector position (3D)
    'target_position': np.array([x, y, z]),
    
    # Target end-effector orientation (quaternion)
    'target_orientation': np.array([qx, qy, qz, qw]),
    
    # Gripper command (continuous from 0 to 1)
    'gripper_command': float,  # 0: open, 1: close
    
    # Tool change command
    'tool_change': int | None  # ID of tool to change to, or None to keep current
}
```

### Step Function
```python
# Execute one step in the environment
observation, reward, done, info = env.step(action)

# observation: dict - The new observation after taking the action
# reward: float - Reward for this step
# done: bool - Whether the episode has ended
# info: dict - Additional information (debugging, metrics, etc.)
```

## Task Management

### Task Structure
```python
# Internal task representation
task = {
    'task_id': int,           # Unique identifier for the task
    'task_type': str,         # Type of task ('reach', 'grasp', 'release', etc.)
    'target_id': int,         # ID of target object
    'target_pose': np.array,  # 7D target pose (position + quaternion)
    'time_limit': float,      # Maximum time allowed (in seconds)
    'completed': bool,        # Whether task is completed
    'prerequisites': List[int] # List of task_ids that must be completed first
}
```

### Task Types
- `reach`: Move end effector to target pose
- `grasp`: Close gripper to grasp target object
- `release`: Open gripper to release held object
- `tool_change`: Change to specified tool
- `composite`: Group of subtasks to be executed in sequence

## Reward Structure

### Rewards
- **Task Completion**: +1.0 for successfully completing a task
- **Progress Reward**: Small positive reward for making progress toward task completion

### Penalties
- **Collision**: -0.1 for each collision with unintended objects
- **Time Penalty**: -0.01 per timestep (encourages efficiency)
- **Invalid Action**: -0.05 for invalid actions (e.g., trying to grasp air)
- **Dropped Object**: -0.5 for dropping an object outside designated areas
- **Tool Change**: -0.1 for changing tools
- **Energy Usage**: -0.01 * (motor_power ** 2) for each motor, where motor_power is the power consumption of the motor (in watts)

## Example Usage

```python
import numpy as np
from car_repair_env import CarRepairEnv

# Initialize environment
env = CarRepairEnv()
obs, info = env.reset()

done = False
total_reward = 0

while not done:
    # Example policy: Move toward target with some noise
    action = {
        'target_pose': obs['target_pose'] + np.random.normal(0, 0.01, 7),
        'gripper_command': 1.0 if 'grasp' in info['current_task_type'] else 0.0,
        'tool_change': None
    }
    
    # Take a step
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    if done:
        print(f"Episode finished with total reward: {total_reward}")
        break

env.close()
```

## Error Handling

### Common Exceptions
- `InvalidActionError`: Raised when an invalid action is provided
- `TaskPrerequisiteError`: Raised when trying to execute a task before its prerequisites are met
- `PhysicsError`: Raised when the physics simulation encounters an unrecoverable error

### Recovery
- Most errors can be recovered from by calling `env.reset()`
- The environment maintains an internal state that can be queried for debugging
