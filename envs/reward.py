# ==============================================================================
# REWARD COMPUTATION - Soft Robotic Tentacle RL
# ==============================================================================
"""
Reward function definitions for the tentacle environment.

This module provides flexible reward computation based on multiple components:
- Distance to target (negative, to encourage reaching)
- Success bonus (reward for reaching target)
- Smoothness penalty (penalize jerky actions)
- Energy penalty (penalize excessive control)
- Alive bonus (small reward per step)

USAGE:
------
    from envs.reward import compute_reward
    
    reward = compute_reward(
        env=env,
        action=action,
        distance=distance_to_target,
        is_success=reached_target,
        config=reward_config
    )

CURRENT MODEL: spiral_5link.xml (5 joints)
FUTURE MODEL:  tentacle.xml (10 joints)
"""

import numpy as np
from typing import Optional, Dict, Any


def compute_reward(
    env,
    action: np.ndarray,
    distance: Optional[float] = None,
    is_success: Optional[bool] = None,
    config: Optional[Dict] = None,
) -> float:
    """
    Compute total reward for current step.
    
    This is the main reward function combining multiple reward components.
    
    Args:
        env: TentacleEnv instance (for accessing state, config, etc.)
        action: Current action taken [shape: (5,) for spiral_5link, (10,) for tentacle.xml]
                # TODO: TENTACLE.XML - Will accept shape (10,) instead
        distance: Distance to target (if None, computed from env)
        is_success: Whether target is reached (if None, computed from env)
        config: Reward config dict (if None, uses env.config['reward'])
    
    Returns:
        total_reward: Sum of all reward components
    
    Reward Equation:
    ----------------
    R = w_dist * r_dist + w_succ * r_succ + w_smooth * r_smooth + w_energy * r_energy + w_alive * r_alive
    
    Where:
        w_dist:    Distance weight (encourage getting closer)
        w_succ:    Success weight (bonus for reaching)
        w_smooth:  Smoothness weight (penalize jerky motion)
        w_energy:  Energy weight (penalize large commands)
        w_alive:   Alive weight (small step reward)
    """
    
    # Get config if not provided
    if config is None:
        config = env.config['reward']
    
    # Get distance if not provided
    if distance is None:
        distance = env._compute_distance_to_target()
    
    # Get success flag if not provided
    if is_success is None:
        is_success = distance < env.success_threshold
    
    total_reward = 0.0
    
    # =========================================================================
    # DISTANCE REWARD - Encourage getting closer to target
    # =========================================================================
    # Negative distance: moving closer = more reward
    # Formula: -distance * weight
    distance_reward = distance_reward_fn(
        distance=distance,
        weight=config.get('weight_distance', 1.0)
    )
    total_reward += distance_reward
    
    # =========================================================================
    # SUCCESS BONUS - Large reward for reaching target
    # =========================================================================
    if is_success:
        success_bonus = success_reward_fn(
            bonus=config.get('bonus_success', 100.0),
            weight=config.get('weight_success', 10.0)
        )
        total_reward += success_bonus
    
    # =========================================================================
    # SMOOTHNESS PENALTY - Penalize jerky/jittery actions
    # =========================================================================
    if env.prev_action is not None:
        smoothness_penalty = smoothness_penalty_fn(
            action=action,
            prev_action=env.prev_action,
            weight=config.get('weight_smoothness', 0.1)
        )
        total_reward += smoothness_penalty
    
    # =========================================================================
    # ENERGY PENALTY - Penalize excessive control effort
    # =========================================================================
    energy_penalty = energy_penalty_fn(
        action=action,
        weight=config.get('weight_energy', 0.01)
    )
    total_reward += energy_penalty
    
    # =========================================================================
    # ALIVE BONUS - Small reward just for surviving each step
    # =========================================================================
    # (Useful to encourage longer episodes in some cases)
    if config.get('weight_alive', 0.0) > 0:
        alive_bonus = config.get('weight_alive', 0.0) * 1.0
        total_reward += alive_bonus
    
    return float(total_reward)


# ==============================================================================
# INDIVIDUAL REWARD COMPONENTS
# ==============================================================================

def distance_reward_fn(distance: float, weight: float = 1.0) -> float:
    """
    Distance-based reward: negative distance scaled by weight.
    
    This encourages the agent to minimize distance to target.
    
    Args:
        distance: Euclidean distance from tip to target
        weight: Scaling weight for this reward component
    
    Returns:
        Reward value (negative, to be minimized)
    
    Formula:
        r_dist = -distance * weight
    
    Example:
        distance=0.5m, weight=1.0 → reward = -0.5
        distance=0.1m, weight=1.0 → reward = -0.1 (better!)
    """
    return -distance * weight


def success_reward_fn(bonus: float = 100.0, weight: float = 1.0) -> float:
    """
    Success bonus: large reward when target is reached.
    
    Args:
        bonus: Base bonus amount (in reward units)
        weight: Scaling weight for this component
    
    Returns:
        Success reward value (large and positive)
    
    Formula:
        r_success = bonus * weight
    
    Example:
        bonus=100, weight=10 → reward = 1000 (strong incentive!)
    """
    return bonus * weight


def smoothness_penalty_fn(
    action: np.ndarray,
    prev_action: np.ndarray,
    weight: float = 0.1,
) -> float:
    """
    Smoothness penalty: penalize rapid action changes (jerk/jitter).
    
    Smooth motion is more realistic and energy-efficient for soft robots.
    Large sudden changes between consecutive actions are penalized.
    
    Args:
        action: Current action [shape: (5,) for spiral_5link, (10,) for tentacle.xml]
                # TODO: TENTACLE.XML - Will handle shape (10,)
        prev_action: Previous action (same shape)
        weight: Penalty weight
    
    Returns:
        Penalty value (negative)
    
    Formula:
        action_diff = sum((action - prev_action)^2)
        r_smooth = -action_diff * weight
    
    Example:
        Small change: action_diff=0.01, weight=0.1 → penalty = -0.001 (minor)
        Large change: action_diff=1.0,  weight=0.1 → penalty = -0.1  (significant)
    """
    action_diff = np.sum(np.square(action - prev_action))
    return -action_diff * weight


def energy_penalty_fn(action: np.ndarray, weight: float = 0.01) -> float:
    """
    Energy penalty: penalize large control signals.
    
    Encourages the agent to use minimal control effort, similar to real robots
    with limited battery or power.
    
    Args:
        action: Control signal [shape: (5,) for spiral_5link, (10,) for tentacle.xml]
                # TODO: TENTACLE.XML - Will handle shape (10,)
        weight: Penalty weight
    
    Returns:
        Energy penalty (negative)
    
    Formula:
        energy = sum(action^2)
        r_energy = -energy * weight
    
    Example:
        Small action: energy=0.01, weight=0.01 → penalty = -0.0001 (negligible)
        Large action: energy=1.0,  weight=0.01 → penalty = -0.01  (noticeable)
    """
    energy = np.sum(np.square(action))
    return -energy * weight


def velocity_penalty_fn(
    joint_velocities: np.ndarray,
    weight: float = 0.001,
) -> float:
    """
    Velocity penalty: penalize high joint velocities.
    
    Encourages smooth, slower motion (useful for some tasks).
    
    Args:
        joint_velocities: Array of joint velocities [shape: (5,) for spiral_5link, (10,) for tentacle.xml]
                          # TODO: TENTACLE.XML - Will handle shape (10,)
        weight: Penalty weight
    
    Returns:
        Velocity penalty (negative)
    
    Formula:
        velocity_sq = sum(qvel^2)
        r_vel = -velocity_sq * weight
    """
    velocity_sq = np.sum(np.square(joint_velocities))
    return -velocity_sq * weight


def custom_distance_metric(
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    metric: str = 'euclidean',
) -> float:
    """
    Compute distance using different metrics.
    
    Args:
        tip_pos: End effector position [x, y, z]
        target_pos: Target position [x, y, z]
        metric: Distance metric ('euclidean', 'manhattan', 'chebyshev')
    
    Returns:
        Distance value
    """
    diff = tip_pos - target_pos
    
    if metric == 'euclidean':
        return np.linalg.norm(diff)
    elif metric == 'manhattan':
        return np.sum(np.abs(diff))
    elif metric == 'chebyshev':
        return np.max(np.abs(diff))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ==============================================================================
# REWARD SHAPING (Advanced)
# ==============================================================================

def shaped_distance_reward(
    distance: float,
    max_distance: float = 2.0,
    weight: float = 1.0,
) -> float:
    """
    Shaped distance reward using normalization.
    
    Normalizes distance to [0, 1] range for more stable learning.
    
    Args:
        distance: Current distance to target
        max_distance: Maximum possible distance (for normalization)
        weight: Reward weight
    
    Returns:
        Shaped reward
    
    Formula:
        normalized_dist = min(distance / max_distance, 1.0)
        r = -normalized_dist * weight
    """
    normalized_dist = min(distance / max_distance, 1.0)
    return -normalized_dist * weight


def exponential_distance_reward(
    distance: float,
    scale: float = 1.0,
    weight: float = 1.0,
) -> float:
    """
    Exponential distance reward: greater reward for being very close.
    
    Provides stronger gradient near the goal.
    
    Args:
        distance: Distance to target
        scale: Exponential scale parameter
        weight: Reward weight
    
    Returns:
        Exponential reward
    
    Formula:
        r = (exp(-scale * distance) - 1) * weight
    """
    return (np.exp(-scale * distance) - 1.0) * weight


# ==============================================================================
# REWARD VALIDATION & DEBUGGING
# ==============================================================================

def validate_reward_config(config: Dict) -> bool:
    """
    Validate reward configuration dictionary.
    
    Checks that all required keys are present and values are reasonable.
    """
    required_keys = [
        'weight_distance',
        'weight_success',
        'bonus_success',
        'weight_smoothness',
        'weight_energy',
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"Warning: Missing reward config key: {key}")
            return False
    
    # Check weights are non-negative
    for key in required_keys:
        if config[key] < 0:
            print(f"Warning: Negative weight for {key}: {config[key]}")
            return False
    
    return True


def print_reward_breakdown(
    reward: float,
    distance_component: float,
    success_component: float,
    smoothness_component: float,
    energy_component: float,
) -> None:
    """Pretty-print reward breakdown for debugging."""
    print("\n" + "="*60)
    print("REWARD BREAKDOWN")
    print("="*60)
    print(f"  Distance Component:    {distance_component:>10.4f}")
    print(f"  Success Component:     {success_component:>10.4f}")
    print(f"  Smoothness Component:  {smoothness_component:>10.4f}")
    print(f"  Energy Component:      {energy_component:>10.4f}")
    print("-"*60)
    print(f"  TOTAL REWARD:          {reward:>10.4f}")
    print("="*60 + "\n")


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    """Test reward functions"""
    print("Testing reward functions...")
    
    # Test distance reward
    dist_reward = distance_reward_fn(0.5, weight=1.0)
    print(f"Distance reward (d=0.5): {dist_reward:.4f}")
    
    # Test success reward
    succ_reward = success_reward_fn(bonus=100.0, weight=10.0)
    print(f"Success reward: {succ_reward:.4f}")
    
    # Test smoothness penalty
    action = np.array([0.5, 0.3, 0.2, 0.1, 0.0])
    prev_action = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    smooth_penalty = smoothness_penalty_fn(action, prev_action, weight=0.1)
    print(f"Smoothness penalty: {smooth_penalty:.4f}")
    
    # Test energy penalty
    energy_penalty = energy_penalty_fn(action, weight=0.01)
    print(f"Energy penalty: {energy_penalty:.4f}")
    
    print("\n✅ All reward functions working!")
