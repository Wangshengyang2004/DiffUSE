import numpy as np
import mujoco
import mujoco.viewer
import cv2
from mujoco_data_generator import CrazyflieSimulator, generate_trajectory, generate_racing_trajectory
import argparse
import sys
import os
import time

class DataCollectionVisualizer:
    def __init__(self, args):
        """Initialize the visualizer with simulation parameters."""
        self.args = args
        self.simulator = CrazyflieSimulator(args)
        self.viewer = None
        self.trajectory_points = []
        self.max_trace_points = 100
        
        # Initialize the viewer
        self.viewer = mujoco.viewer.launch_passive(self.simulator.model, self.simulator.data)
        
        # Add visualization elements
        self.add_visualization_elements()
        
        # Initialize trajectory generator
        if args.trajectory_type == 'racing':
            self.trajectory_generator = generate_racing_trajectory
        else:
            self.trajectory_generator = generate_trajectory
    
    def add_visualization_elements(self):
        """Add visualization elements to the scene."""
        # Store trajectory points for visualization
        self.trajectory_points = []
        self.current_point_index = 0
        
        # We'll reuse existing sites if they exist, or create new ones if needed
        n_existing_sites = self.simulator.model.site_pos.shape[0]
        self.trace_sites = list(range(n_existing_sites))
    
    def update_visualization(self, position):
        """Update visualization elements."""
        # Add current position to trajectory points
        self.trajectory_points.append(position.copy())
        if len(self.trajectory_points) > self.max_trace_points:
            self.trajectory_points.pop(0)
        
        # Update site positions for trajectory visualization
        for i, point in enumerate(self.trajectory_points):
            if i < len(self.trace_sites):
                self.simulator.data.site_xpos[i] = point
    
    def run(self):
        """Run the visualization."""
        t = 0
        dt = self.simulator.model.opt.timestep
        
        try:
            while t < 10.0:  # Run for 10 seconds
                # Generate target state
                target_state = self.trajectory_generator(t, self.args)
                
                # Step simulation
                self.simulator.step(target_state)
                
                # Update visualization
                current_pos = self.simulator.data.qpos[:3]
                self.update_visualization(current_pos)
                
                # Update viewer
                self.viewer.sync()
                
                # Increment time
                t += dt
                
                # Small sleep to control visualization speed
                time.sleep(dt)
        
        finally:
            # Cleanup
            self.viewer.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize data collection')
    # Model and simulation parameters
    parser.add_argument('--model_path', type=str, default='src/diffuse/sim/assets/racing_quad.xml',
                      help='Path to the MuJoCo model XML file')
    parser.add_argument('--dt', type=float, default=0.01,
                      help='Simulation timestep')
    parser.add_argument('--camera_resolution', type=tuple, default=(64, 64),
                      help='Camera resolution (height, width)')
    parser.add_argument('--depth_resolution', type=tuple, default=(64, 64),
                      help='Depth resolution (height, width)')
    parser.add_argument('--simple_mode', action='store_true',
                      help='Use simplified simulation without complex rendering')
    
    # Trajectory parameters
    parser.add_argument('--trajectory_type', type=str, default='racing',
                      help='Type of trajectory to generate')
    parser.add_argument('--max_velocity', type=float, default=10.0,
                      help='Maximum velocity')
    parser.add_argument('--max_angular_velocity', type=float, default=8.0,
                      help='Maximum angular velocity')
    parser.add_argument('--trajectory_frequency', type=float, default=1.0,
                      help='Trajectory frequency for racing mode')
    parser.add_argument('--racing_aggressiveness', type=float, default=1.2,
                      help='Aggressiveness factor for racing trajectories')
    parser.add_argument('--min_height', type=float, default=0.3,
                      help='Minimum height for trajectories')
    
    args = parser.parse_args()
    visualizer = DataCollectionVisualizer(args)
    visualizer.run()

if __name__ == '__main__':
    main() 