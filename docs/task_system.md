# Task System

This document describes the task system for the car repair simulation environment, including objectives, rewards, and task progression.

## Task Structure

### 1. Task Definition
Each task consists of:
- **Objective**: The goal to be achieved (e.g., "Remove battery terminal")
- **Target**: The specific object to interact with (e.g., "negative_terminal")
- **Success Criteria**: Conditions that must be met to complete the task
- **Time Limit**: Maximum time allowed for the task

Rewards are flat because there is no point in awarding some tasks more than others.
Penalties are given when the arm is hitting any object it is not supposed to hit, and when any object that is not supposed to be moved is significantly moved.

### 2. Task Types

#### A. Basic Interaction Tasks
1. **Reach**: Move end effector to target AABB
   - Example: Position tool near battery terminal
   - Success: End effector enters target volume

2. **Grasp/Release**: Pick up or release an object
   - Example: Pick up a screwdriver
   - Success: Object is secured/released by end effector. When released, it must be released at the target location. The target location is either the car or the tooling system.

3. **Insert/Remove**: Place or remove a component
   - Example: Remove oil filter
   - Success: Component is properly seated/removed

#### B. Composite Tasks
1. **Tool Change**: Swap end effectors
   - Example: Switch from gripper to screwdriver
   - Success: New tool is properly attached

2. **Sequential Tasks**: Multiple steps in sequence
   - Example: 1) Remove oil cap, 2) Add oil, 3) Replace cap
   - Success: All steps completed in order

## Environment Integration

The task system is tightly integrated with the [Environment API](./environment_api.md), which handles the low-level simulation and control. Key aspects of this integration include:

- **Task Execution**: The environment manages the execution of tasks in the specified order, enforcing prerequisites and time limits.
- **State Tracking**: The environment maintains the current task state and provides feedback through the observation space.
- **Reward Calculation**: Rewards and penalties are calculated based on task progress and interactions.

For detailed information about the environment's observation space, action space, and reward structure, see the [Environment API documentation](./environment_api.md).

## Task Implementation

### Task Definition

Tasks are defined using a structured dictionary format that the environment can interpret. This format allows for flexible task specification while maintaining a clear structure for both simple and complex repair procedures.

#### Core Task Structure
```python
{
    'task_id': int,               # Unique identifier for the task
    'task_type': str,             # Type of task (see below)
    'target_id': int,             # ID of target object
    'target_position': np.array,   # [x, y, z] in meters
    'target_orientation': np.array, # [qx, qy, qz, qw] quaternion
    'time_limit': float,          # Maximum time allowed (seconds)
    'prerequisites': List[int],   # Required completed task IDs
    'tool_requirement': int,      # Required tool ID (-1 for none)
    'success_distance': float,    # Max distance for success (meters)
    'success_angle': float        # Max angle difference (radians)
}
```

#### Task Types

1. **Reach**
   - Move end effector to target position/orientation
   - Parameters: position, orientation, success thresholds
   
2. **Grasp**
   - Close gripper to pick up an object
   - Requires: gripper tool, valid graspable target
   - Parameters: grasp force, max approach angle
   
3. **Release**
   - Open gripper to release held object
   - Can specify target position/orientation for placement
   
4. **Tool Change**
   - Switch to specified tool
   - Must be performed at tool stand location
   - Parameters: tool_id
   
5. **Wait**
   - Wait for specified duration
   - Used for timing-dependent operations
   - Parameters: duration (seconds)

### Example Task Sequence (Machine Code)
```python
# Example task sequence for removing battery terminal
tasks = [
    {
        'task_id': 0,
        'task_type': 'reach',
        'target_id': 100,  # ID for battery terminal
        'target_pose': np.array([0.5, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0]),
        'time_limit': 10.0,
        'prerequisites': []
    },
    {
        'task_id': 1,
        'task_type': 'grasp',
        'target_id': 100,  # Same target as previous step
        'target_pose': np.array([0.5, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0]),
        'time_limit': 5.0,
        'prerequisites': [0]  # Must complete task 0 first
    },
    # ... more tasks
]

# Task completion is signaled by the environment when the end effector
# reaches the target pose with appropriate conditions (e.g., gripper state)
```

## Task Progression

### Execution Flow
1. Tasks are executed in the order specified by their `task_id`
2. A task becomes active when:
   - All prerequisites are completed
   - Required tool is equipped
   - No higher-priority tasks are pending

### Success Conditions
- **Reach**: End effector within `success_distance` and `success_angle` of target
- **Grasp/Release**: Successful gripper operation with proper contact
- **Tool Change**: Tool successfully attached/detected
- **Wait**: Specified duration elapsed

### Error Handling
- Task fails if time limit is exceeded
- Invalid tool usage results in immediate failure
- Collisions may trigger task failure based on severity

### Episode Termination
An episode ends when:
- All tasks are completed successfully
- Any task fails irrecoverably
- Global time limit is reached
- Maximum number of steps is exceeded
