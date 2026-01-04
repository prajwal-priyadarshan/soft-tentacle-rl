# ==============================================================================
# TERMINATION CONDITIONS - Soft Robotic Tentacle RL
# ==============================================================================
"""
Episode termination condition definitions.

This module provides functions to check when an episode should end.
An episode can end in three ways:
1. SUCCESS: Target reached (terminated = True)
2. FAILURE: Collision or instability (terminated = True)
3. TIMEOUT: Max steps exceeded (truncated = True)

TERMINOLOGY:
-----------
- terminated: Episode ended naturally (success or failure)
- truncated: Episode cut short (timeout)

Both are standard Gymnasium conventions.

CURRENT MODEL: spiral_5link.xml (5 joints, planar)
FUTURE MODEL:  tentacle.xml (10 joints, 3D motion)

USAGE:
------
    from envs.termination import check_termination
    
    terminated, truncated = check_termination(env)
    
    # Or individual checks
    if is_success(env):
        print("Target reached!")
    
    if is_timeout(env):
        print("Episode timeout!")
"""

import numpy as np
from typing import Tuple, Optional, Dict


def check_termination(env, config: Optional[Dict] = None) -> Tuple[bool, bool]:
    """
    Check all termination conditions for the current episode.
    
    This is the main termination checking function used by the environment.
    It combines multiple termination criteria into a single decision.
    
    Args:
        env: TentacleEnv instance
        config: Episode config (if None, uses env.config['episode'])
    
    Returns:
        terminated: True if episode ended naturally (success/failure)
        truncated: True if episode was cut short (timeout)
    
    Decision Logic:
    ---------------
    1. Check SUCCESS: Target reached → terminated=True
    2. Check FAILURE: Invalid state detected → terminated=True
    3. Check TIMEOUT: Max steps exceeded → truncated=True
    4. Otherwise: Both False (episode continues)
    
    Note:
    -----
    - An episode can be EITHER terminated OR truncated, NOT both
    - Both can be False if episode should continue
    """
    
    # Get config if not provided
    if config is None:
        config = env.config['episode']
    
    terminated = False
    truncated = False
    
    # =========================================================================
    # SUCCESS CHECK - Target reached
    # =========================================================================
    if config.get('terminate_on_success', True):
        if is_success(env):
            terminated = True
            return terminated, truncated
    
    # =========================================================================
    # FAILURE CHECK - Invalid state or collision
    # =========================================================================
    if config.get('terminate_on_failure', True):
        if is_failure(env):
            terminated = True
            return terminated, truncated
    
    # =========================================================================
    # TIMEOUT CHECK - Max steps exceeded
    # =========================================================================
    if is_timeout(env):
        truncated = True
        return terminated, truncated
    
    # No termination condition met
    return terminated, truncated


# ==============================================================================
# SUCCESS CONDITION
# ==============================================================================

def is_success(env) -> bool:
    """
    Check if target has been reached (success condition).
    
    Success is defined as the tip being within success_threshold distance
    of the target for sufficient consecutive steps.
    
    Args:
        env: TentacleEnv instance
    
    Returns:
        True if target is reached
    
    Metrics:
    --------
    - success_threshold: Distance threshold (e.g., 0.05m = 5cm)
    - Instant success: No hysteresis required
    
    Example:
        threshold = 0.05m
        distance = 0.03m → SUCCESS! (closer than threshold)
        distance = 0.07m → NOT success (farther than threshold)
    """
    distance = get_distance_to_target(env)
    success = distance < env.success_threshold
    
    return bool(success)


def is_success_with_hysteresis(
    env,
    threshold: float = 0.05,
    steps_required: int = 5,
    history: Optional[list] = None,
) -> bool:
    """
    Check success with hysteresis (stability requirement).
    
    Requires the agent to maintain success for multiple consecutive steps.
    Prevents early termination due to brief oscillations.
    
    Args:
        env: TentacleEnv instance
        threshold: Distance threshold for success
        steps_required: Number of consecutive steps needed
        history: List tracking success state (internal use)
    
    Returns:
        True if success maintained for required steps
    
    Note:
    -----
    This is more strict than simple distance check.
    Useful if target is difficult to reach precisely.
    """
    # This would need state tracking - not used in basic implementation
    # but provided for advanced usage
    pass


# ==============================================================================
# FAILURE CONDITIONS
# ==============================================================================

def is_failure(env) -> bool:
    """
    Check if episode should terminate due to failure.
    
    Multiple failure conditions:
    1. Self-collision (tentacle hitting itself)
    2. Joint limit violation (unrealistic strain)
    3. Extreme joint velocities (instability)
    4. Tip going below ground (falling)
    
    Args:
        env: TentacleEnv instance
    
    Returns:
        True if failure condition detected
    """
    
    # =========================================================================
    # COLLISION DETECTION
    # =========================================================================
    if _check_self_collision(env):
        return True
    
    # =========================================================================
    # JOINT LIMIT VIOLATION
    # =========================================================================
    if _check_joint_limits(env):
        return True
    
    # =========================================================================
    # EXCESSIVE JOINT VELOCITIES
    # =========================================================================
    if _check_velocity_limits(env):
        return True
    
    # =========================================================================
    # TIP BELOW GROUND (falling)
    # =========================================================================
    if _check_tip_below_ground(env):
        return True
    
    return False


def _check_self_collision(env) -> bool:
    """
    Check for self-collision (tentacle hitting itself).
    
    FUTURE: Implement collision detection
    - Use MuJoCo's contact detection
    - Check for contacts between non-adjacent links
    
    Current implementation: Not active (returns False)
    """
    # TODO: TENTACLE.XML - Implement when 3D model is ready
    # Soft robots can self-intersect - need to detect and penalize
    
    # Placeholder: Check MuJoCo contacts
    # for contact in env.data.contact:
    #     geom1 = contact.geom1
    #     geom2 = contact.geom2
    #     # Check if contact is self-collision (same kinematic chain)
    #     if is_self_collision(geom1, geom2):
    #         return True
    
    return False


def _check_joint_limits(env) -> bool:
    """
    Check if any joint is beyond its limits (with tolerance).
    
    MuJoCo enforces limits by default, but we can detect
    repeated near-limit violations as instability.
    
    Returns:
        True if joint is critically beyond limits
    """
    # Check if any position is beyond limits by > 5%
    for i, q in enumerate(env.data.qpos):
        lower = env.model.jnt_range[i, 0]
        upper = env.model.jnt_range[i, 1]
        
        if q < lower - 0.05 * (upper - lower) or q > upper + 0.05 * (upper - lower):
            return True
    
    return False


def _check_velocity_limits(env, max_velocity: float = 20.0) -> bool:
    """
    Check if any joint velocity is excessively high.
    
    Very high velocities indicate instability or control failure.
    
    Args:
        env: TentacleEnv instance
        max_velocity: Maximum acceptable velocity (rad/s)
    
    Returns:
        True if any joint exceeds velocity limit
    """
    max_vel = np.max(np.abs(env.data.qvel))
    
    if max_vel > max_velocity:
        return True
    
    return False


def _check_tip_below_ground(env, ground_level: float = 0.0) -> bool:
    """
    Check if tip has fallen below ground plane.
    
    The ground is at z=0 in the simulation.
    
    Args:
        env: TentacleEnv instance
        ground_level: Z-coordinate of ground plane
    
    Returns:
        True if tip is below ground
    """
    tip_z = env._get_tip_position()[2]
    
    if tip_z < ground_level:
        return True
    
    return False


# ==============================================================================
# TIMEOUT CONDITION
# ==============================================================================

def is_timeout(env) -> bool:
    """
    Check if episode has exceeded maximum step limit.
    
    Standard truncation condition to prevent infinite episodes.
    
    Args:
        env: TentacleEnv instance
    
    Returns:
        True if max_steps exceeded
    
    Note:
    -----
    - truncated = True (not terminated)
    - Agent should learn to solve task before timeout
    - Typically max_steps = 500-1000 for reaching tasks
    """
    return env.current_step >= env.max_steps


# ==============================================================================
# DISTANCE-BASED TERMINATION
# ==============================================================================

def get_distance_to_target(env) -> float:
    """
    Get Euclidean distance from tip to target.
    
    Used by success condition check.
    
    Returns:
        Distance in meters
    """
    tip_pos = env._get_tip_position()
    distance = np.linalg.norm(tip_pos - env.target_pos)
    return distance


def check_approach_to_target(env, improvement_threshold: float = 0.01) -> bool:
    """
    Check if agent is making progress toward target.
    
    Can be used to detect stuck agents and terminate episode.
    
    Args:
        env: TentacleEnv instance
        improvement_threshold: Minimum distance improvement per step
    
    Returns:
        True if agent is NOT making progress (failure condition)
    
    Implementation Note:
    --------------------
    This requires tracking previous distance in the environment.
    Not implemented in basic version but provided for advanced usage.
    """
    # This would require storing previous distance in env state
    # Example logic:
    # current_dist = get_distance_to_target(env)
    # if hasattr(env, '_prev_distance'):
    #     improvement = env._prev_distance - current_dist
    #     if improvement < improvement_threshold:
    #         return True  # No progress
    # env._prev_distance = current_dist
    
    return False


# ==============================================================================
# TERMINATION UTILITIES
# ==============================================================================

def get_termination_reason(env) -> str:
    """
    Get human-readable reason for episode termination.
    
    Args:
        env: TentacleEnv instance
    
    Returns:
        String describing why episode ended
    """
    if is_success(env):
        return "SUCCESS: Target reached!"
    
    if is_timeout(env):
        return "TIMEOUT: Max steps exceeded"
    
    if _check_self_collision(env):
        return "FAILURE: Self-collision detected"
    
    if _check_joint_limits(env):
        return "FAILURE: Joint limit exceeded"
    
    if _check_velocity_limits(env):
        return "FAILURE: Excessive joint velocity"
    
    if _check_tip_below_ground(env):
        return "FAILURE: Tip below ground"
    
    return "CONTINUING: Episode in progress"


def should_render_termination(terminated: bool, truncated: bool) -> bool:
    """
    Determine if termination event should trigger special rendering.
    
    Returns:
        True if episode ended (should show result)
    """
    return terminated or truncated


# ==============================================================================
# TERMINATION LOGGING
# ==============================================================================

def log_episode_termination(
    env,
    terminated: bool,
    truncated: bool,
    episode_num: int,
    total_reward: float,
) -> None:
    """
    Log episode termination information for analysis.
    
    Args:
        env: TentacleEnv instance
        terminated: Whether episode terminated naturally
        truncated: Whether episode timed out
        episode_num: Episode number (for logging)
        total_reward: Cumulative reward for episode
    """
    reason = get_termination_reason(env)
    distance = get_distance_to_target(env)
    
    print(f"\nEpisode {episode_num} ended:")
    print(f"  Reason: {reason}")
    print(f"  Final distance to target: {distance:.4f}m")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Steps taken: {env.current_step}/{env.max_steps}")
    
    if terminated:
        print("  Status: EPISODE TERMINATED")
    elif truncated:
        print("  Status: EPISODE TRUNCATED (timeout)")


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    """Test termination functions"""
    print("Testing termination module...")
    print("Note: Run this from TentacleEnv instance context")
    
    print("\nTermination condition types:")
    print("  1. SUCCESS - Target reached (terminated=True)")
    print("  2. FAILURE - Invalid state (terminated=True)")
    print("  3. TIMEOUT - Max steps exceeded (truncated=True)")
    print("  4. CONTINUE - None of above (both=False)")
    
    print("\n✅ Termination module loaded successfully!")
