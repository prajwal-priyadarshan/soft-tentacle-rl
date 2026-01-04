# ==============================================================================
# UTILITY FUNCTIONS - Soft Robotic Tentacle RL
# ==============================================================================
"""
Helper functions for the tentacle environment.

This module provides utility functions for:
- Configuration loading and validation
- Observation normalization and preprocessing
- Distance calculations and geometric transforms
- State space utilities
- Logging and debugging

CURRENT MODEL: spiral_5link.xml (5 joints, planar)
FUTURE MODEL:  tentacle.xml (10 joints, 3D motion)

USAGE:
------
    from envs.utils import load_config, normalize_obs, compute_distance
    
    # Load configuration
    config = load_config('configs/env.yaml')
    
    # Normalize observation
    obs_norm = normalize_obs(obs, obs_mean, obs_std)
    
    # Compute distance
    dist = compute_distance(tip_pos, target_pos, metric='euclidean')
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import warnings


# ==============================================================================
# CONFIGURATION UTILITIES
# ==============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
                    Can be absolute or relative to project root
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    
    Example:
        config = load_config('configs/env.yaml')
        num_actuators = config['action']['num_actuators']
    """
    # Handle relative paths
    if not os.path.isabs(config_path):
        # Try relative to project root
        project_root = _get_project_root()
        config_path = os.path.join(project_root, config_path)
    
    # Check file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config: {e}")
    
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Deep merge override config into base config.
    
    Override values take precedence over base values.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Overrides to apply
    
    Returns:
        Merged configuration
    
    Example:
        base = load_config('configs/env.yaml')
        override = {'action': {'num_actuators': 10}}
        config = merge_configs(base, override)
        # Now config['action']['num_actuators'] = 10
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict) -> bool:
    """
    Validate that config has all required keys.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, False otherwise
    
    Prints warnings for missing keys.
    """
    required_sections = [
        'model',
        'action',
        'observation',
        'target',
        'episode',
        'reward',
        'render',
        'randomization',
    ]
    
    valid = True
    for section in required_sections:
        if section not in config:
            warnings.warn(f"Missing config section: {section}")
            valid = False
    
    return valid


def print_config(config: Dict, indent: int = 0) -> None:
    """
    Pretty-print configuration dictionary.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level (for recursion)
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def get_model_info(config: Dict) -> Dict[str, Any]:
    """
    Extract model information from config.
    
    Returns:
        Dictionary with model specs:
        - num_actuators: Number of motors
        - model_type: 'spiral_5link' or 'tentacle_3d'
        - num_segments: Number of segments
        - num_joints: Number of joints
    
    CURRENT (spiral_5link.xml): 5 joints, 1 per segment
    TODO: TENTACLE.XML - 10 joints, 2 per segment
    """
    num_actuators = config['action']['num_actuators']
    
    model_info = {
        'num_actuators': num_actuators,
        'num_segments': 5,  # Always 5 segments in both models
        'num_joints': num_actuators,
    }
    
    # Determine model type
    if num_actuators == 5:
        model_info['model_type'] = 'spiral_5link'
        model_info['motion_type'] = 'planar'
        model_info['dof_per_segment'] = 1
    elif num_actuators == 10:
        model_info['model_type'] = 'tentacle_3d'
        model_info['motion_type'] = '3D'
        model_info['dof_per_segment'] = 2
    else:
        raise ValueError(f"Unknown number of actuators: {num_actuators}")
    
    return model_info


# ==============================================================================
# OBSERVATION NORMALIZATION
# ==============================================================================

class RunningNormalizer:
    """
    Compute running mean and std for observation normalization.
    
    Tracks statistics over many steps for adaptive normalization.
    Useful for training stability in RL.
    
    Example:
        normalizer = RunningNormalizer(obs_dim=22)
        
        for obs in observations_stream:
            normalizer.update(obs)
            obs_norm = normalizer.normalize(obs)
    """
    
    def __init__(self, obs_dim: int, clip_range: float = 5.0, eps: float = 1e-8):
        """
        Initialize running normalizer.
        
        Args:
            obs_dim: Observation dimension
            clip_range: Clip normalized values to [-clip_range, clip_range]
            eps: Small value to avoid division by zero
        """
        self.obs_dim = obs_dim
        self.clip_range = clip_range
        self.eps = eps
        
        # Running statistics
        self.mean = np.zeros(obs_dim)
        self.var = np.ones(obs_dim)
        self.count = 0
    
    def update(self, obs: np.ndarray):
        """Update running mean and variance."""
        obs = np.asarray(obs)
        batch_mean = obs.mean(axis=0) if obs.ndim > 1 else obs
        batch_var = obs.var(axis=0) if obs.ndim > 1 else 0
        batch_count = obs.shape[0] if obs.ndim > 1 else 1
        
        self._update_mean_var(batch_mean, batch_var, batch_count)
    
    def _update_mean_var(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Internal update using Welford's online algorithm."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running stats."""
        obs = np.asarray(obs)
        normalized = (obs - self.mean) / np.sqrt(self.var + self.eps)
        return np.clip(normalized, -self.clip_range, self.clip_range).astype(np.float32)
    
    def denormalize(self, obs_norm: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        obs_norm = np.asarray(obs_norm)
        return (obs_norm * np.sqrt(self.var + self.eps) + self.mean).astype(np.float32)


def normalize_observation(
    obs: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    clip: float = 10.0,
) -> np.ndarray:
    """
    Normalize observation to zero mean, unit variance.
    
    Args:
        obs: Observation to normalize
        mean: Running mean (if None, assumes obs is pre-centered)
        std: Running std (if None, assumes obs is pre-scaled)
        clip: Clip normalized values to [-clip, +clip]
    
    Returns:
        Normalized observation
    
    Formula:
        obs_norm = (obs - mean) / std
        obs_norm = clip(obs_norm, -clip, +clip)
    """
    obs = np.asarray(obs, dtype=np.float32)
    
    if mean is not None:
        obs = obs - mean
    
    if std is not None:
        obs = obs / (std + 1e-8)
    
    obs = np.clip(obs, -clip, clip)
    
    return obs.astype(np.float32)


def normalize_action(
    action: np.ndarray,
    low: float = -1.0,
    high: float = 1.0,
) -> np.ndarray:
    """
    Clip action to valid range.
    
    Args:
        action: Action vector
        low: Minimum action value
        high: Maximum action value
    
    Returns:
        Clipped action
    """
    return np.clip(action, low, high).astype(np.float32)


# ==============================================================================
# GEOMETRIC UTILITIES
# ==============================================================================

def compute_distance(
    point1: np.ndarray,
    point2: np.ndarray,
    metric: str = 'euclidean',
) -> float:
    """
    Compute distance between two 3D points.
    
    Args:
        point1: First point [x, y, z]
        point2: Second point [x, y, z]
        metric: Distance metric
                - 'euclidean': L2 norm (default)
                - 'manhattan': L1 norm
                - 'chebyshev': L-infinity norm
    
    Returns:
        Distance value (scalar)
    
    Example:
        tip = np.array([0.1, 0.2, 0.3])
        target = np.array([0.5, 0.3, 0.8])
        dist = compute_distance(tip, target)
        # dist ≈ 0.527
    """
    diff = np.asarray(point1) - np.asarray(point2)
    
    if metric == 'euclidean':
        return float(np.linalg.norm(diff))
    elif metric == 'manhattan':
        return float(np.sum(np.abs(diff)))
    elif metric == 'chebyshev':
        return float(np.max(np.abs(diff)))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> float:
    """
    Compute shortest distance from point to a line segment.
    
    Useful for collision detection and complex reward shaping.
    
    Args:
        point: 3D point
        line_start: Start of line segment
        line_end: End of line segment
    
    Returns:
        Minimum distance
    """
    point = np.asarray(point)
    line_start = np.asarray(line_start)
    line_end = np.asarray(line_end)
    
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0:
        return float(np.linalg.norm(point - line_start))
    
    t = np.dot(point - line_start, line_vec) / (line_len ** 2)
    t = np.clip(t, 0, 1)
    
    closest_point = line_start + t * line_vec
    return float(np.linalg.norm(point - closest_point))


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π] range.
    
    Args:
        angle: Angle in radians
    
    Returns:
        Normalized angle
    """
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def angles_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """
    Convert Euler angles to 3x3 rotation matrix.
    
    Args:
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians)
    
    Returns:
        3x3 rotation matrix
    """
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation (ZYX order)
    return Rz @ Ry @ Rx


# ==============================================================================
# STATE UTILITIES
# ==============================================================================

def stack_observations(
    observations: List[np.ndarray],
    stack_size: int = 4,
) -> np.ndarray:
    """
    Stack multiple observations for temporal context.
    
    Useful for policies that need history (e.g., velocity estimation).
    
    Args:
        observations: List of observation vectors
        stack_size: Number of observations to stack
    
    Returns:
        Stacked observation vector
    
    Example:
        obs_history = [obs_t-3, obs_t-2, obs_t-1, obs_t]
        stacked = stack_observations(obs_history, stack_size=4)
        # Returns concatenated vector of all 4 observations
    """
    # Take only last stack_size observations
    obs_to_stack = observations[-stack_size:]
    
    # Pad with first observation if not enough history
    while len(obs_to_stack) < stack_size:
        obs_to_stack.insert(0, obs_to_stack[0])
    
    return np.concatenate(obs_to_stack).astype(np.float32)


def compute_returns(
    rewards: List[float],
    gamma: float = 0.99,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute cumulative discounted returns.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        normalize: Whether to normalize returns
    
    Returns:
        Array of returns (one per timestep)
    """
    returns = []
    G = 0
    
    # Compute returns backwards
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = np.array(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


# ==============================================================================
# SAMPLING UTILITIES
# ==============================================================================

def sample_uniform_in_box(
    bounds: Dict[str, Tuple[float, float]],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample a 3D point uniformly in a bounding box.
    
    Args:
        bounds: Dictionary with 'x', 'y', 'z' keys
                Each contains (min, max) tuple
        rng: Random number generator (if None, uses default)
    
    Returns:
        Sampled 3D point
    
    Example:
        bounds = {
            'x': (-0.5, 0.5),
            'y': (-0.3, 0.3),
            'z': (0.3, 1.2),
        }
        point = sample_uniform_in_box(bounds)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    x = rng.uniform(bounds['x'][0], bounds['x'][1])
    y = rng.uniform(bounds['y'][0], bounds['y'][1])
    z = rng.uniform(bounds['z'][0], bounds['z'][1])
    
    return np.array([x, y, z])


# ==============================================================================
# LOGGING UTILITIES
# ==============================================================================

def print_env_info(env) -> None:
    """Print information about environment."""
    print("\n" + "="*60)
    print("ENVIRONMENT INFORMATION")
    print("="*60)
    print(f"Model: {env.config['model']['xml_path']}")
    print(f"Number of joints: {env.num_joints}")
    print(f"Number of actuators: {env.num_actuators}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Max episode steps: {env.max_steps}")
    print(f"Success threshold: {env.success_threshold}m")
    print("="*60 + "\n")


def format_episode_stats(stats: Dict[str, float]) -> str:
    """Format episode statistics for printing."""
    lines = []
    lines.append("Episode Statistics:")
    
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


# ==============================================================================
# FILE UTILITIES
# ==============================================================================

def _get_project_root() -> str:
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # utils.py is in envs/, so go up one level
    project_root = os.path.dirname(current_dir)
    return project_root


def ensure_dir_exists(dir_path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(dir_path, exist_ok=True)


def save_numpy_array(array: np.ndarray, file_path: str) -> None:
    """Save numpy array to file."""
    ensure_dir_exists(os.path.dirname(file_path))
    np.save(file_path, array)


def load_numpy_array(file_path: str) -> np.ndarray:
    """Load numpy array from file."""
    return np.load(file_path)


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test config loading
    config_path = 'configs/env.yaml'
    try:
        config = load_config(config_path)
        print(f"✅ Config loaded: {len(config)} sections")
    except Exception as e:
        print(f"⚠️  Config loading: {e}")
    
    # Test distance computation
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    dist = compute_distance(p1, p2)
    print(f"✅ Distance: {dist:.4f} (expected 1.0)")
    
    # Test running normalizer
    normalizer = RunningNormalizer(obs_dim=10)
    obs = np.random.randn(10)
    normalizer.update(obs)
    obs_norm = normalizer.normalize(obs)
    print(f"✅ Normalization: obs shape {obs.shape} -> {obs_norm.shape}")
    
    print("\n✅ All utility functions working!")
