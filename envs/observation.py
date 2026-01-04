# ==============================================================================
# OBSERVATION BUILDING - Soft Robotic Tentacle RL
# ==============================================================================
"""
Observation space definition and building functions.

This module provides functions to construct observation vectors from the
environment state. Observations are fed to the RL agent for decision making.

OBSERVATION COMPONENTS:
-----------------------
- Joint positions (qpos): Current angle of each joint
- Joint velocities (qvel): Angular velocity of each joint
- Tip position: 3D coordinates of end effector
- Target position: 3D coordinates of target goal
- Distance to target: Scalar distance metric
- Previous action: Last action taken (for smoothness)

CURRENT MODEL (spiral_5link.xml):
---------------------------------
Total observation dimension: 22
- Joint positions:  5
- Joint velocities: 5
- Tip position:     3
- Target position:  3
- Distance:         1
- Previous action:  5

FUTURE MODEL (tentacle.xml):
-----------------------------
Total observation dimension: 37 (when updated)
- Joint positions:  10
- Joint velocities: 10
- Tip position:     3
- Target position:  3
- Distance:         1
- Previous action:  10

USAGE:
------
    from envs.observation import build_observation
    
    obs = build_observation(
        env=env,
        config=observation_config
    )
    
    # Or get individual components
    joint_pos = get_joint_positions(env)
    tip_pos = get_tip_position(env)
    distance = get_distance_to_target(env)
"""

import numpy as np
import mujoco
from typing import Dict, Any, Optional


def build_observation(
    env,
    config: Optional[Dict] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Build complete observation vector from environment state.
    
    This is the main observation building function used by the environment.
    It concatenates multiple observation components based on configuration.
    
    Args:
        env: TentacleEnv instance with current state
        config: Observation configuration dict (if None, uses env.config['observation'])
        normalize: Whether to normalize observations to [-1, 1]
    
    Returns:
        observation: Observation vector as numpy array (float32)
                     CURRENT (spiral_5link.xml): shape=(22,)
                     TODO: TENTACLE.XML - Will be shape=(37,)
    
    Structure:
    ----------
    The returned observation vector is organized as:
    [joint_pos_0, joint_pos_1, ..., joint_pos_N,
     joint_vel_0, joint_vel_1, ..., joint_vel_N,
     tip_x, tip_y, tip_z,
     target_x, target_y, target_z,
     distance_to_target,
     prev_action_0, prev_action_1, ..., prev_action_N]
    
    Example (spiral_5link.xml):
    [j1_pos, j2_pos, j3_pos, j4_pos, j5_pos,          # 5
     j1_vel, j2_vel, j3_vel, j4_vel, j5_vel,          # 5
     tip_x, tip_y, tip_z,                             # 3
     target_x, target_y, target_z,                    # 3
     distance,                                         # 1
     act_0, act_1, act_2, act_3, act_4]               # 5
    Total: 22 components
    """
    
    # Get config if not provided
    if config is None:
        config = env.config['observation']
    
    obs_parts = []
    
    # =========================================================================
    # JOINT POSITIONS
    # =========================================================================
    # CURRENT (spiral_5link.xml): 5 values [j1, j2, j3, j4, j5]
    # TODO: TENTACLE.XML - Will be 10 values [j1_pitch, j1_yaw, j2_pitch, j2_yaw, ...]
    if config.get('include_joint_pos', True):
        joint_pos = get_joint_positions(env)
        obs_parts.append(joint_pos)
    
    # =========================================================================
    # JOINT VELOCITIES
    # =========================================================================
    # CURRENT (spiral_5link.xml): 5 values
    # TODO: TENTACLE.XML - Will be 10 values
    if config.get('include_joint_vel', True):
        joint_vel = get_joint_velocities(env)
        obs_parts.append(joint_vel)
    
    # =========================================================================
    # TIP (END EFFECTOR) POSITION
    # =========================================================================
    # Always 3 values (x, y, z) for any model
    if config.get('include_tip_pos', True):
        tip_pos = get_tip_position(env)
        obs_parts.append(tip_pos)
    
    # =========================================================================
    # TARGET POSITION
    # =========================================================================
    # Always 3 values (x, y, z)
    if config.get('include_target_pos', True):
        target_pos = get_target_position(env)
        obs_parts.append(target_pos)
    
    # =========================================================================
    # DISTANCE TO TARGET
    # =========================================================================
    # Single scalar value
    if config.get('include_distance', True):
        distance = get_distance_to_target(env)
        obs_parts.append(np.array([distance]))
    
    # =========================================================================
    # PREVIOUS ACTION
    # =========================================================================
    # CURRENT (spiral_5link.xml): 5 values
    # TODO: TENTACLE.XML - Will be 10 values
    if config.get('include_prev_action', True):
        if env.prev_action is not None:
            prev_action = env.prev_action.copy()
        else:
            prev_action = np.zeros(env.config['action']['num_actuators'])
        obs_parts.append(prev_action)
    
    # =========================================================================
    # CONCATENATE ALL PARTS
    # =========================================================================
    observation = np.concatenate(obs_parts).astype(np.float32)
    
    # =========================================================================
    # OPTIONAL: NORMALIZE OBSERVATIONS
    # =========================================================================
    if normalize:
        observation = normalize_observation(observation)
    
    return observation


# ==============================================================================
# INDIVIDUAL OBSERVATION COMPONENTS
# ==============================================================================

def get_joint_positions(env) -> np.ndarray:
    """
    Get current joint positions (angles).
    
    Returns:
        Joint position array (qpos)
        CURRENT (spiral_5link.xml): shape=(5,)
        TODO: TENTACLE.XML - shape=(10,)
    
    Joint order for spiral_5link.xml: [j1, j2, j3, j4, j5]
    Joint order for tentacle.xml:     [j1_pitch, j1_yaw, j2_pitch, j2_yaw, ..., j5_yaw]
    
    Units: Radians
    Range: Typically [-π/3, π/3] based on XML (±60°)
    """
    return env.data.qpos.copy().astype(np.float32)


def get_joint_velocities(env) -> np.ndarray:
    """
    Get current joint velocities (angular velocities).
    
    Returns:
        Joint velocity array (qvel)
        CURRENT (spiral_5link.xml): shape=(5,)
        TODO: TENTACLE.XML - shape=(10,)
    
    Units: Radians per second
    Range: Unbounded in theory, but typically [-10, 10] rad/s
    """
    return env.data.qvel.copy().astype(np.float32)


def get_tip_position(env) -> np.ndarray:
    """
    Get the 3D position of the tentacle tip (end effector).
    
    The tip is the distal end of link5 (the final segment).
    
    Returns:
        Tip position [x, y, z]
        Units: Meters
    
    Implementation:
    ---------------
    1. Get the base position of link5 from xpos array
    2. Get rotation matrix of link5 from xmat array
    3. Add offset for link length (0.25m) along local z-axis
    
    Formula:
        tip_pos = link5_pos + R_link5 @ [0, 0, 0.25]
    
    Where:
        link5_pos: World position of link5 body
        R_link5:   3x3 rotation matrix of link5
        0.25:      Link length in local z-axis
    
    Works for both spiral_5link.xml and tentacle.xml since both have link5 as tip.
    """
    # Get the body index for link5
    tip_body_id = env.tip_body_id
    
    # Get base position of link5 in world frame
    link5_pos = env.data.xpos[tip_body_id].copy()
    
    # Get rotation matrix of link5
    # xmat is stored as 9 elements, reshape to 3x3
    link5_rot = env.data.xmat[tip_body_id].reshape(3, 3)
    
    # Offset for link length (0.25m along local z-axis)
    link_length = 0.25  # From XML: fromto="0 0 0  0 0 0.25"
    local_offset = np.array([0.0, 0.0, link_length])
    
    # Transform offset to world frame
    world_offset = link5_rot @ local_offset
    
    # Get tip position in world frame
    tip_pos = link5_pos + world_offset
    
    return tip_pos.astype(np.float32)


def get_target_position(env) -> np.ndarray:
    """
    Get the 3D position of the target.
    
    Returns:
        Target position [x, y, z]
        Units: Meters
    """
    return env.target_pos.copy().astype(np.float32)


def get_distance_to_target(env) -> float:
    """
    Get Euclidean distance from tip to target.
    
    Returns:
        Distance value (scalar, positive)
        Units: Meters
    
    Formula:
        distance = ||tip_pos - target_pos||_2
                 = sqrt((tip_x - target_x)^2 + (tip_y - target_y)^2 + (tip_z - target_z)^2)
    """
    tip_pos = get_tip_position(env)
    target_pos = get_target_position(env)
    distance = np.linalg.norm(tip_pos - target_pos)
    return float(distance)


def get_body_position(env, body_name: str) -> np.ndarray:
    """
    Get position of a named body in the model.
    
    Args:
        env: TentacleEnv instance
        body_name: Name of body (e.g., 'link1', 'link5', 'base')
    
    Returns:
        Body position [x, y, z]
    
    Example:
        link1_pos = get_body_position(env, 'link1')
        link5_pos = get_body_position(env, 'link5')
    """
    body_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    return env.data.xpos[body_id].copy().astype(np.float32)


# ==============================================================================
# OBSERVATION NORMALIZATION & PREPROCESSING
# ==============================================================================

def normalize_observation(obs: np.ndarray) -> np.ndarray:
    """
    Normalize observation to [-1, 1] range.
    
    Simple scaling normalization (not adaptive/running stats).
    Assumes observations are already in reasonable ranges from the environment.
    
    Args:
        obs: Raw observation vector
    
    Returns:
        Normalized observation (clipped to [-1, 1])
    """
    # Simple clip normalization
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


def standardize_observation(
    obs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Standardize observation using running mean/std.
    
    Useful for training stability (zero-mean, unit-variance).
    
    Args:
        obs: Raw observation
        mean: Running mean (computed over training history)
        std: Running standard deviation
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Standardized observation: (obs - mean) / (std + eps)
    """
    return ((obs - mean) / (std + eps)).astype(np.float32)


def clip_observation(
    obs: np.ndarray,
    min_val: float = -10.0,
    max_val: float = 10.0,
) -> np.ndarray:
    """
    Clip observation to bounded range.
    
    Prevents extreme values from affecting training.
    
    Args:
        obs: Raw observation
        min_val: Minimum clipping value
        max_val: Maximum clipping value
    
    Returns:
        Clipped observation
    """
    return np.clip(obs, min_val, max_val).astype(np.float32)


# ==============================================================================
# OBSERVATION UTILITIES
# ==============================================================================

def get_observation_dim(config: Dict, num_joints: int) -> int:
    """
    Calculate total observation dimension based on config.
    
    Args:
        config: Observation config dictionary
        num_joints: Number of joints in the model
                   CURRENT: 5 for spiral_5link.xml
                   TODO: TENTACLE.XML - 10
    
    Returns:
        Total observation dimension
    """
    dim = 0
    
    if config.get('include_joint_pos', True):
        dim += num_joints
    
    if config.get('include_joint_vel', True):
        dim += num_joints
    
    if config.get('include_tip_pos', True):
        dim += 3
    
    if config.get('include_target_pos', True):
        dim += 3
    
    if config.get('include_distance', True):
        dim += 1
    
    if config.get('include_prev_action', True):
        dim += num_joints  # Same as action dim
    
    return dim


def get_observation_names(config: Dict, num_joints: int) -> list:
    """
    Get list of observation component names.
    
    Useful for debugging and understanding the observation vector.
    
    Args:
        config: Observation config
        num_joints: Number of joints
    
    Returns:
        List of observation names in order
    """
    names = []
    
    if config.get('include_joint_pos', True):
        names.extend([f'j{i+1}_pos' for i in range(num_joints)])
    
    if config.get('include_joint_vel', True):
        names.extend([f'j{i+1}_vel' for i in range(num_joints)])
    
    if config.get('include_tip_pos', True):
        names.extend(['tip_x', 'tip_y', 'tip_z'])
    
    if config.get('include_target_pos', True):
        names.extend(['target_x', 'target_y', 'target_z'])
    
    if config.get('include_distance', True):
        names.append('distance')
    
    if config.get('include_prev_action', True):
        names.extend([f'prev_action_{i}' for i in range(num_joints)])
    
    return names


def print_observation_breakdown(obs: np.ndarray, names: list) -> None:
    """Print observation vector for debugging."""
    print("\n" + "="*60)
    print("OBSERVATION BREAKDOWN")
    print("="*60)
    
    for i, (name, value) in enumerate(zip(names, obs)):
        print(f"  [{i:2d}] {name:20s} = {value:>10.4f}")
    
    print("="*60 + "\n")


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    """Test observation functions"""
    print("Testing observation module...")
    print("Note: Run this from TentacleEnv instance context")
    
    # Test observation component names
    config = {
        'include_joint_pos': True,
        'include_joint_vel': True,
        'include_tip_pos': True,
        'include_target_pos': True,
        'include_distance': True,
        'include_prev_action': True,
    }
    
    num_joints = 5  # For spiral_5link.xml
    obs_dim = get_observation_dim(config, num_joints)
    obs_names = get_observation_names(config, num_joints)
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Observation components: {len(obs_names)}")
    print(f"Component names: {obs_names[:5]}... (showing first 5)")
    
    print("\n✅ Observation module loaded successfully!")
