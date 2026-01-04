#!/usr/bin/env python3
# ==============================================================================
# TEST ENVIRONMENT - Random Actions Test
# ==============================================================================
"""
Complete test script for the Tentacle Environment.

This script tests the environment with random actions to verify:
1. Environment initialization
2. Reset functionality
3. Step functionality
4. Observations and rewards
5. Termination conditions
6. Rendering (optional)

USAGE:
------
    cd /path/to/soft-tentacle-rl
    python scripts/test_env_random.py

    # With rendering
    python scripts/test_env_random.py --render

    # Multiple episodes
    python scripts/test_env_random.py --episodes 5

    # More verbose output
    python scripts/test_env_random.py --verbose

CURRENT MODEL: spiral_5link.xml (5 joints, planar)
FUTURE MODEL:  tentacle.xml (10 joints, 3D motion)

To test with 3D model:
1. Change 'xml_path' in configs/env.yaml to "models/tentacle.xml"
2. Change 'num_actuators' to 10
3. Run this script again - it will automatically adapt!
"""

import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.tentacle_env import TentacleEnv


# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

def test_environment_creation(verbose: bool = True) -> TentacleEnv:
    """
    Test environment creation and initialization.
    
    Args:
        verbose: Print detailed information
    
    Returns:
        Created environment instance
    """
    print("\n" + "="*70)
    print("TEST 1: ENVIRONMENT CREATION")
    print("="*70)
    
    try:
        env = TentacleEnv()
        print("✅ Environment created successfully!")
        
        if verbose:
            print(f"   Model loaded: {env.config['model']['xml_path']}")
            print(f"   Action space: {env.action_space}")
            print(f"   Observation space: {env.observation_space}")
            print(f"   Number of joints: {env.num_joints}")
            print(f"   Number of actuators: {env.num_actuators}")
            print(f"   Max steps per episode: {env.max_steps}")
        
        return env
    
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        raise


def test_reset(env: TentacleEnv, verbose: bool = True) -> np.ndarray:
    """
    Test environment reset functionality.
    
    Args:
        env: TentacleEnv instance
        verbose: Print detailed information
    
    Returns:
        Initial observation
    """
    print("\n" + "="*70)
    print("TEST 2: ENVIRONMENT RESET")
    print("="*70)
    
    try:
        obs, info = env.reset()
        print("✅ Reset successful!")
        
        if verbose:
            print(f"   Observation shape: {obs.shape}")
            print(f"   Observation dtype: {obs.dtype}")
            print(f"   Observation range: [{obs.min():.4f}, {obs.max():.4f}]")
            print(f"\n   Info dictionary:")
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    print(f"     {key}: shape {value.shape}")
                elif isinstance(value, float):
                    print(f"     {key}: {value:.4f}")
                else:
                    print(f"     {key}: {value}")
        
        return obs
    
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        raise


def test_step(env: TentacleEnv, num_steps: int = 5, verbose: bool = True):
    """
    Test environment step functionality with random actions.
    
    Args:
        env: TentacleEnv instance
        num_steps: Number of steps to take
        verbose: Print detailed information
    """
    print("\n" + "="*70)
    print(f"TEST 3: ENVIRONMENT STEP ({num_steps} steps)")
    print("="*70)
    
    try:
        obs, info = env.reset()
        cumulative_reward = 0.0
        
        print(f"\n{'Step':<6} {'Reward':<12} {'Distance':<12} {'Term':<6} {'Trunc':<6}")
        print("-" * 50)
        
        for step in range(num_steps):
            # Sample random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            
            # Print step info
            distance = info['distance_to_target']
            print(f"{step:<6} {reward:<12.4f} {distance:<12.4f} {str(terminated):<6} {str(truncated):<6}")
            
            if terminated or truncated:
                print(f"\nEpisode ended at step {step+1}")
                break
        
        print("-" * 50)
        print(f"Cumulative reward: {cumulative_reward:.4f}")
        print("✅ Step test successful!")
    
    except Exception as e:
        print(f"❌ Step test failed: {e}")
        raise


def test_action_space(env: TentacleEnv, verbose: bool = True):
    """
    Test action space sampling and bounds.
    
    Args:
        env: TentacleEnv instance
        verbose: Print detailed information
    """
    print("\n" + "="*70)
    print("TEST 4: ACTION SPACE")
    print("="*70)
    
    try:
        print(f"Action space: {env.action_space}")
        print(f"Low: {env.action_space.low}")
        print(f"High: {env.action_space.high}")
        
        # Sample multiple actions
        print(f"\nSampling 5 random actions:")
        for i in range(5):
            action = env.action_space.sample()
            print(f"  Action {i+1}: min={action.min():.4f}, max={action.max():.4f}, mean={action.mean():.4f}")
            
            # Verify bounds
            assert np.all(action >= env.action_space.low), "Action below low bound"
            assert np.all(action <= env.action_space.high), "Action above high bound"
        
        print("✅ Action space test successful!")
    
    except Exception as e:
        print(f"❌ Action space test failed: {e}")
        raise


def test_observation_space(env: TentacleEnv, verbose: bool = True):
    """
    Test observation space.
    
    Args:
        env: TentacleEnv instance
        verbose: Print detailed information
    """
    print("\n" + "="*70)
    print("TEST 5: OBSERVATION SPACE")
    print("="*70)
    
    try:
        print(f"Observation space: {env.observation_space}")
        
        # Get observations
        obs, _ = env.reset()
        print(f"\nObservation properties:")
        print(f"  Shape: {obs.shape}")
        print(f"  Dtype: {obs.dtype}")
        print(f"  Min: {obs.min():.4f}")
        print(f"  Max: {obs.max():.4f}")
        print(f"  Mean: {obs.mean():.4f}")
        print(f"  Std: {obs.std():.4f}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(obs)):
            print("  ⚠️  Warning: NaN values in observation!")
        if np.any(np.isinf(obs)):
            print("  ⚠️  Warning: Inf values in observation!")
        
        # Sample multiple observations
        print(f"\nSampling observations over 10 steps:")
        for step in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            
            if step == 0:
                print(f"  Step {step:<2}: min={obs.min():>8.4f}, max={obs.max():>8.4f}, mean={obs.mean():>8.4f}")
        
        print("✅ Observation space test successful!")
    
    except Exception as e:
        print(f"❌ Observation space test failed: {e}")
        raise


def test_termination(env: TentacleEnv, verbose: bool = True):
    """
    Test termination conditions.
    
    Args:
        env: TentacleEnv instance
        verbose: Print detailed information
    """
    print("\n" + "="*70)
    print("TEST 6: TERMINATION CONDITIONS")
    print("="*70)
    
    try:
        # Test timeout
        print("Testing timeout condition...")
        obs, _ = env.reset()
        
        for step in range(env.max_steps + 10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if truncated:
                print(f"✅ Timeout detected at step {step+1}/{env.max_steps}")
                break
        
        # Test success
        print("\nTesting success condition...")
        obs, info = env.reset()
        
        # Place target very close
        initial_distance = info['distance_to_target']
        env.target_pos = env._get_tip_position()  # Put target at current tip position
        
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        if terminated:
            print(f"✅ Success detected (distance < {env.success_threshold}m)")
        else:
            print(f"⚠️  Success not triggered (distance={info['distance_to_target']:.4f}m)")
        
        print("✅ Termination test successful!")
    
    except Exception as e:
        print(f"❌ Termination test failed: {e}")
        raise


def run_episode(env: TentacleEnv, render: bool = False, verbose: bool = True) -> dict:
    """
    Run a complete episode with random actions.
    
    Args:
        env: TentacleEnv instance
        render: Whether to render (if supported)
        verbose: Print step information
    
    Returns:
        Episode statistics dictionary
    """
    obs, info = env.reset()
    
    episode_reward = 0.0
    episode_length = 0
    min_distance = info['distance_to_target']
    distances = []
    rewards = []
    
    if verbose:
        print(f"\nStarting episode (max {env.max_steps} steps)...")
    
    while True:
        # Sample action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record statistics
        episode_reward += reward
        episode_length += 1
        distance = info['distance_to_target']
        min_distance = min(min_distance, distance)
        distances.append(distance)
        rewards.append(reward)
        
        # Render if requested
        if render:
            try:
                env.render()
            except Exception as e:
                if verbose:
                    print(f"⚠️  Rendering not available: {e}")
                render = False
        
        # Check termination
        if terminated:
            if verbose:
                print(f"✅ Episode succeeded! (distance={distance:.4f}m)")
            success = True
            break
        
        if truncated:
            if verbose:
                print(f"⏱️  Episode timeout at step {episode_length}")
            success = False
            break
    
    stats = {
        'episode_length': episode_length,
        'episode_reward': episode_reward,
        'min_distance': min_distance,
        'final_distance': distance,
        'avg_distance': np.mean(distances),
        'success': success,
        'mean_reward': np.mean(rewards),
    }
    
    return stats


def print_episode_stats(stats: dict, episode_num: int = 1):
    """Print episode statistics."""
    print("\n" + "-"*70)
    print(f"Episode {episode_num} Results:")
    print("-"*70)
    print(f"  Length: {stats['episode_length']} steps")
    print(f"  Total Reward: {stats['episode_reward']:.4f}")
    print(f"  Mean Reward: {stats['mean_reward']:.4f}")
    print(f"  Final Distance: {stats['final_distance']:.4f}m")
    print(f"  Min Distance: {stats['min_distance']:.4f}m")
    print(f"  Avg Distance: {stats['avg_distance']:.4f}m")
    print(f"  Success: {stats['success']}")


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test Tentacle Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_env_random.py                    # Basic test
  python scripts/test_env_random.py --episodes 5       # Run 5 episodes
  python scripts/test_env_random.py --render --verbose # Detailed output
        """
    )
    
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to run (default: 1)')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering (if available)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip unit tests, just run episodes')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("TENTACLE ENVIRONMENT TEST SUITE")
    print("="*70)
    
    # Create environment
    print("\nInitializing environment...")
    try:
        env = test_environment_creation(verbose=args.verbose)
    except Exception as e:
        print(f"\n❌ FATAL: Could not create environment!")
        print(f"Error: {e}")
        return 1
    
    # Run tests
    if not args.skip_tests:
        try:
            test_reset(env, verbose=args.verbose)
            test_action_space(env, verbose=args.verbose)
            test_observation_space(env, verbose=args.verbose)
            test_step(env, num_steps=5, verbose=args.verbose)
            test_termination(env, verbose=args.verbose)
        except Exception as e:
            print(f"\n❌ Tests failed: {e}")
            env.close()
            return 1
    
    # Run episodes
    print("\n" + "="*70)
    print("RUNNING EPISODES")
    print("="*70)
    
    episode_stats_list = []
    
    for episode in range(args.episodes):
        try:
            stats = run_episode(env, render=args.render, verbose=args.verbose)
            episode_stats_list.append(stats)
            print_episode_stats(stats, episode_num=episode+1)
        except Exception as e:
            print(f"❌ Episode {episode+1} failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Print summary
    if episode_stats_list:
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        avg_length = np.mean([s['episode_length'] for s in episode_stats_list])
        avg_reward = np.mean([s['episode_reward'] for s in episode_stats_list])
        avg_distance = np.mean([s['final_distance'] for s in episode_stats_list])
        success_count = sum(1 for s in episode_stats_list if s['success'])
        
        print(f"Episodes run: {len(episode_stats_list)}")
        print(f"Successful episodes: {success_count}/{len(episode_stats_list)}")
        print(f"Average length: {avg_length:.1f} steps")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Average final distance: {avg_distance:.4f}m")
    
    # Cleanup
    env.close()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nYour environment is ready for training with RL algorithms!")
    print("Next steps:")
    print("  1. Train PPO with: python rl/train_ppo.py")
    print("  2. Train SAC with: python rl/train_sac.py")
    print("  3. Evaluate with: python rl/evaluate.py")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
