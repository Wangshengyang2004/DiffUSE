#!/usr/bin/env python
"""
Generate UAV dataset using MuJoCo-based Crazyflie simulator.
Supports different drone configurations and trajectory types.
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from diffuse.sim.mujoco_data_generator import generate_dataset

def generate_racing_trajectory(t, max_velocity=10.0, frequency=1.0):
    """Generate racing-style trajectory with aggressive maneuvers."""
    # Base figure-8 pattern
    x = max_velocity * 0.5 * np.sin(2 * np.pi * frequency * t)
    y = max_velocity * 0.25 * np.sin(4 * np.pi * frequency * t)
    z = max_velocity * 0.15 * np.sin(6 * np.pi * frequency * t) + 1.0  # Keep some minimum height
    
    # Add high-frequency components for more aggressive motion
    x += max_velocity * 0.1 * np.sin(8 * np.pi * frequency * t)
    y += max_velocity * 0.1 * np.sin(10 * np.pi * frequency * t)
    z += max_velocity * 0.05 * np.sin(12 * np.pi * frequency * t)
    
    return np.array([x, y, z])

def main():
    parser = argparse.ArgumentParser(description="Generate MuJoCo-based UAV dataset")
    
    # Simulation mode and model
    parser.add_argument('--drone_type', type=str, default='racing',
                      choices=['minimal', 'standard', 'heavy', 'racing', 'camera'],
                      help='Type of drone to simulate')
    parser.add_argument('--model_path', type=str,
                      help='Path to MuJoCo XML model file (if not specified, chosen based on type)')
    
    # Simulation parameters
    parser.add_argument('--dt', type=float, default=0.01,
                      help='Time step in seconds')
    parser.add_argument('--trajectory_type', type=str, default='racing',
                      choices=['random', 'circle', 'figure8', 'racing', 'hover'],
                      help='Type of trajectory to generate')
    parser.add_argument('--max_velocity', type=float, default=10.0,
                      help='Maximum linear velocity')
    parser.add_argument('--max_angular_velocity', type=float, default=8.0,
                      help='Maximum angular velocity')
    parser.add_argument('--trajectory_frequency', type=float, default=1.0,
                      help='Base frequency for trajectory generation')
    
    # Racing-specific parameters
    parser.add_argument('--racing_aggressiveness', type=float, default=1.0,
                      help='Multiplier for racing maneuver aggressiveness')
    parser.add_argument('--min_height', type=float, default=0.5,
                      help='Minimum height for safety')
    
    # Dataset parameters
    parser.add_argument('--output_dir', type=str, default='data/simulated',
                      help='Output directory for dataset')
    parser.add_argument('--num_trajectories', type=int, default=100,
                      help='Number of trajectories to generate')
    parser.add_argument('--trajectory_duration', type=float, default=10.0,
                      help='Duration of each trajectory in seconds')
    parser.add_argument('--splits', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                      help='Dataset splits (train, val, test)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Noise parameters
    parser.add_argument('--noise_levels', type=float, nargs=4, default=[0.02, 0.05, 0.1, 0.05],
                      help='Noise levels for IMU, camera, depth, position')
    
    args = parser.parse_args()
    
    # Set model path based on drone type if not specified
    if args.model_path is None:
        model_map = {
            'minimal': 'src/diffuse/sim/assets/crazyflie_minimal.xml',
            'standard': 'src/diffuse/sim/assets/crazyflie_standard.xml',
            'heavy': 'src/diffuse/sim/assets/heavy_lift_quad.xml',
            'racing': 'src/diffuse/sim/assets/racing_quad.xml',
            'camera': 'src/diffuse/sim/assets/camera_quad.xml'
        }
        args.model_path = model_map[args.drone_type]
    
    # Convert noise levels to dictionary
    args.noise_levels = {
        'imu': args.noise_levels[0],
        'camera': args.noise_levels[1],
        'depth': args.noise_levels[2],
        'position': args.noise_levels[3]
    }
    
    # Print configuration
    print("\nGenerating dataset with the following configuration:")
    print(f"  Drone type: {args.drone_type}")
    print(f"  Model: {args.model_path}")
    print(f"  Timestep: {args.dt}s")
    print(f"  Trajectory type: {args.trajectory_type}")
    print(f"  Max velocity: {args.max_velocity} m/s")
    print(f"  Max angular velocity: {args.max_angular_velocity} rad/s")
    print(f"  Number of trajectories: {args.num_trajectories}")
    print(f"  Duration per trajectory: {args.trajectory_duration}s")
    print(f"  Output directory: {args.output_dir}")
    
    if args.trajectory_type == 'racing':
        print("\nRacing configuration:")
        print(f"  Aggressiveness: {args.racing_aggressiveness}")
        print(f"  Trajectory frequency: {args.trajectory_frequency} Hz")
        print(f"  Minimum height: {args.min_height}m")
    
    print("\nNoise levels:")
    print(f"  IMU: {args.noise_levels['imu']}")
    print(f"  Camera: {args.noise_levels['camera']}")
    print(f"  Depth: {args.noise_levels['depth']}")
    print(f"  Position: {args.noise_levels['position']}")
    
    # Generate dataset
    generate_dataset(args)

if __name__ == "__main__":
    main() 