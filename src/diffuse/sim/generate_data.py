import os
import numpy as np
import argparse
import cv2
from tqdm import tqdm
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from diffuse.models.physics_model import UAVPhysicsModel

class UAVSimulator:
    """
    Simulates UAV flights to generate training data for the diffusion model.
    """
    def __init__(
        self,
        dt=0.01,
        trajectory_type='random',
        environment_size=(10, 10, 5),
        camera_resolution=(64, 64),
        depth_resolution=(64, 64),
        max_velocity=5.0,
        max_angular_velocity=3.0,
        noise_levels={'imu': 0.02, 'camera': 0.05, 'depth': 0.1, 'position': 0.05}
    ):
        """
        Initialize the UAV simulator.
        
        Args:
            dt: Time step in seconds
            trajectory_type: Type of trajectory to generate ('random', 'circle', 'figure8', 'hover')
            environment_size: Size of the environment (x, y, z)
            camera_resolution: Resolution of the camera images
            depth_resolution: Resolution of the depth maps
            max_velocity: Maximum linear velocity
            max_angular_velocity: Maximum angular velocity
            noise_levels: Noise levels for different sensors
        """
        self.dt = dt
        self.trajectory_type = trajectory_type
        self.environment_size = environment_size
        self.camera_resolution = camera_resolution
        self.depth_resolution = depth_resolution
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.noise_levels = noise_levels
        
        # Initialize physics model for state propagation
        self.physics_model = UAVPhysicsModel(
            dt=dt,
            max_velocity=max_velocity,
            max_angular_velocity=max_angular_velocity
        )
        
        # Set up environment (simple colored blocks in 3D space)
        self.environment = self._setup_environment()
    
    def _setup_environment(self):
        """Set up a simple 3D environment with obstacles."""
        # Create some random obstacles in the environment
        num_obstacles = 20
        obstacles = []
        
        for _ in range(num_obstacles):
            # Random position
            x = random.uniform(0, self.environment_size[0])
            y = random.uniform(0, self.environment_size[1])
            z = random.uniform(0, self.environment_size[2])
            
            # Random size
            size_x = random.uniform(0.2, 1.0)
            size_y = random.uniform(0.2, 1.0)
            size_z = random.uniform(0.2, 1.0)
            
            # Random color
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            
            obstacles.append({
                'position': (x, y, z),
                'size': (size_x, size_y, size_z),
                'color': color
            })
        
        return {
            'obstacles': obstacles,
            'size': self.environment_size
        }
    
    def generate_trajectory(self, duration):
        """
        Generate a UAV trajectory.
        
        Args:
            duration: Duration of the trajectory in seconds
            
        Returns:
            timestamps: List of timestamps
            states: List of UAV states
        """
        # Calculate number of time steps
        num_steps = int(duration / self.dt)
        
        # Initialize trajectory
        timestamps = [i * self.dt for i in range(num_steps)]
        states = []
        
        # Initialize state
        if self.trajectory_type == 'hover':
            # Hover in the middle of the environment
            initial_state = np.zeros(12)
            initial_state[0:3] = [self.environment_size[0]/2, self.environment_size[1]/2, self.environment_size[2]/2]
        else:
            # Random initial position
            initial_state = np.zeros(12)
            initial_state[0:3] = [
                random.uniform(0.5, self.environment_size[0]-0.5),
                random.uniform(0.5, self.environment_size[1]-0.5),
                random.uniform(0.5, self.environment_size[2]-0.5)
            ]
        
        states.append(initial_state)
        
        # Generate control inputs for each time step based on trajectory type
        for i in range(1, num_steps):
            if self.trajectory_type == 'random':
                # Random control inputs
                control = np.zeros(4)
                # Thrust to counteract gravity plus some random variation
                control[0] = self.physics_model.mass * self.physics_model.gravity + random.uniform(-2.0, 2.0)
                # Random moments
                control[1:4] = np.random.uniform(-0.5, 0.5, 3)
            
            elif self.trajectory_type == 'circle':
                # Circle trajectory
                t = i * self.dt
                radius = min(self.environment_size[0], self.environment_size[1]) / 3
                center_x = self.environment_size[0] / 2
                center_y = self.environment_size[1] / 2
                center_z = self.environment_size[2] / 2
                
                # Target position on circle
                target_x = center_x + radius * np.cos(t)
                target_y = center_y + radius * np.sin(t)
                target_z = center_z
                
                # Current position
                current_pos = states[-1][0:3]
                
                # Simple PD control to follow trajectory
                kp = 5.0  # Position gain
                kd = 1.0  # Velocity gain
                
                # Calculate desired acceleration
                pos_error = np.array([target_x, target_y, target_z]) - current_pos
                vel_error = -states[-1][6:9]  # Assuming desired velocity is zero for simplicity
                
                desired_accel = kp * pos_error + kd * vel_error
                
                # Convert to control inputs (very simplified)
                control = np.zeros(4)
                # Thrust to counteract gravity plus z acceleration
                control[0] = self.physics_model.mass * (self.physics_model.gravity + desired_accel[2])
                # Roll and pitch to control x,y motion (simplified)
                control[1] = desired_accel[0] * 0.1  # Roll
                control[2] = desired_accel[1] * 0.1  # Pitch
                control[3] = 0.0  # Yaw
            
            elif self.trajectory_type == 'figure8':
                # Figure-8 trajectory
                t = i * self.dt
                radius = min(self.environment_size[0], self.environment_size[1]) / 4
                center_x = self.environment_size[0] / 2
                center_y = self.environment_size[1] / 2
                center_z = self.environment_size[2] / 2
                
                # Target position on figure-8
                target_x = center_x + radius * np.sin(t)
                target_y = center_y + radius * np.sin(t) * np.cos(t)
                target_z = center_z
                
                # Current position
                current_pos = states[-1][0:3]
                
                # Simple PD control to follow trajectory
                kp = 5.0  # Position gain
                kd = 1.0  # Velocity gain
                
                # Calculate desired acceleration
                pos_error = np.array([target_x, target_y, target_z]) - current_pos
                vel_error = -states[-1][6:9]  # Assuming desired velocity is zero for simplicity
                
                desired_accel = kp * pos_error + kd * vel_error
                
                # Convert to control inputs (very simplified)
                control = np.zeros(4)
                # Thrust to counteract gravity plus z acceleration
                control[0] = self.physics_model.mass * (self.physics_model.gravity + desired_accel[2])
                # Roll and pitch to control x,y motion (simplified)
                control[1] = desired_accel[0] * 0.1  # Roll
                control[2] = desired_accel[1] * 0.1  # Pitch
                control[3] = 0.0  # Yaw
            
            elif self.trajectory_type == 'hover':
                # Hover trajectory (maintain position)
                control = np.zeros(4)
                # Thrust to counteract gravity
                control[0] = self.physics_model.mass * self.physics_model.gravity
                # No moments
                control[1:4] = np.zeros(3)
            
            # Propagate state using physics model
            next_state = self.physics_model.simulate_step(
                state=torch.from_numpy(states[-1]).unsqueeze(0).float(), 
                control_inputs=torch.from_numpy(control).unsqueeze(0).float()
            ).squeeze(0).cpu().numpy()
            
            # Ensure UAV stays within environment bounds
            next_state[0:3] = np.clip(next_state[0:3], 0, [self.environment_size[0], self.environment_size[1], self.environment_size[2]])
            
            states.append(next_state)
        
        return timestamps, np.array(states)
    
    def generate_sensor_data(self, states, timestamps):
        """
        Generate sensor readings for a given trajectory.
        
        Args:
            states: Array of UAV states
            timestamps: List of timestamps
            
        Returns:
            sensor_data: Dictionary of sensor readings
        """
        num_steps = len(timestamps)
        
        # Initialize sensor data
        sensor_data = {
            'imu': [],
            'camera': [],
            'depth': [],
            'position': []
        }
        
        # Generate sensor readings for each time step
        for i in range(num_steps):
            state = states[i]
            
            # IMU (accelerometer + gyroscope)
            imu_reading = self._generate_imu_reading(state)
            sensor_data['imu'].append(imu_reading)
            
            # Camera (RGB image)
            camera_reading = self._generate_camera_reading(state)
            sensor_data['camera'].append(camera_reading)
            
            # Depth (depth map)
            depth_reading = self._generate_depth_reading(state)
            sensor_data['depth'].append(depth_reading)
            
            # Position (from Lighthouse V2 or similar)
            position_reading = self._generate_position_reading(state)
            sensor_data['position'].append(position_reading)
        
        return sensor_data
    
    def _generate_imu_reading(self, state):
        """Generate a simulated IMU reading (accelerometer + gyroscope)."""
        # Extract linear and angular velocities
        linear_velocity = state[6:9]
        angular_velocity = state[9:12]
        
        # Calculate linear acceleration (simplified)
        # In a real implementation, we'd convert from world to body frame
        linear_acceleration = np.zeros(3)
        linear_acceleration[2] = -self.physics_model.gravity  # Gravity component
        
        # Combine accelerometer and gyroscope readings
        imu_reading = np.concatenate([linear_acceleration, angular_velocity])
        
        # Add noise
        noise = np.random.normal(0, self.noise_levels['imu'], imu_reading.shape)
        imu_reading = imu_reading + noise
        
        return imu_reading
    
    def _generate_camera_reading(self, state):
        """Generate a simulated camera image."""
        # Create a blank image
        img = np.zeros((self.camera_resolution[0], self.camera_resolution[1], 3), dtype=np.uint8)
        
        # Extract position and orientation
        position = state[0:3]
        orientation = state[3:6]
        
        # Draw a simplified view of the environment
        # This is a very basic implementation - in a real system, 
        # you'd use a proper renderer or simulator like Gazebo/IsaacSim
        
        # Add a ground plane gradient
        for y in range(self.camera_resolution[1]):
            # Calculate intensity based on y coordinate (higher = darker)
            intensity = 255 - int(200 * y / self.camera_resolution[1])
            img[y, :, :] = [intensity, intensity, intensity]
        
        # Add some simple geometric shapes to represent obstacles
        for obstacle in self.environment['obstacles']:
            # Calculate obstacle position relative to UAV
            rel_pos = (
                obstacle['position'][0] - position[0],
                obstacle['position'][1] - position[1],
                obstacle['position'][2] - position[2]
            )
            
            # Very simplified perspective projection
            # In reality, you'd use proper camera projection
            if rel_pos[2] > 0:  # Only render obstacles in front of camera
                proj_x = int(self.camera_resolution[0]/2 + rel_pos[0]/rel_pos[2] * 30)
                proj_y = int(self.camera_resolution[1]/2 + rel_pos[1]/rel_pos[2] * 30)
                
                # Draw a simple circle for the obstacle
                size = int(10 * obstacle['size'][0] / max(0.1, rel_pos[2]))
                if (proj_x >= 0 and proj_x < self.camera_resolution[0] and 
                    proj_y >= 0 and proj_y < self.camera_resolution[1]):
                    cv2.circle(img, (proj_x, proj_y), size, obstacle['color'], -1)
        
        # Add noise
        noise = np.random.normal(0, self.noise_levels['camera'] * 255, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def _generate_depth_reading(self, state):
        """Generate a simulated depth map."""
        # Create a blank depth map
        depth = np.ones((self.depth_resolution[0], self.depth_resolution[1])) * 10.0
        
        # Extract position and orientation
        position = state[0:3]
        orientation = state[3:6]
        
        # Generate a simplified depth map
        # This is a very basic implementation - in a real system,
        # you'd use a proper renderer or simulator
        
        # Add ground plane depth
        for y in range(self.depth_resolution[1]):
            # Calculate depth based on position and orientation
            ground_dist = position[2] / max(0.01, np.cos(orientation[0]))
            row_depth = ground_dist * (1 + 0.1 * y / self.depth_resolution[1])
            depth[y, :] = row_depth
        
        # Add obstacles depth values
        for obstacle in self.environment['obstacles']:
            # Calculate obstacle position relative to UAV
            rel_pos = (
                obstacle['position'][0] - position[0],
                obstacle['position'][1] - position[1],
                obstacle['position'][2] - position[2]
            )
            
            # Distance to obstacle
            dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
            
            # Very simplified perspective projection
            if rel_pos[2] > 0:  # Only render obstacles in front of camera
                proj_x = int(self.depth_resolution[0]/2 + rel_pos[0]/rel_pos[2] * 30)
                proj_y = int(self.depth_resolution[1]/2 + rel_pos[1]/rel_pos[2] * 30)
                
                # Set depth value for obstacle
                size = int(10 * obstacle['size'][0] / max(0.1, rel_pos[2]))
                for y in range(max(0, proj_y-size), min(self.depth_resolution[1], proj_y+size)):
                    for x in range(max(0, proj_x-size), min(self.depth_resolution[0], proj_x+size)):
                        if (x - proj_x)**2 + (y - proj_y)**2 <= size**2:
                            depth[y, x] = min(depth[y, x], dist)
        
        # Add noise
        noise = np.random.normal(0, self.noise_levels['depth'], depth.shape)
        depth = depth + noise
        
        # Clip to reasonable range
        depth = np.clip(depth, 0.1, 10.0)
        
        return depth
    
    def _generate_position_reading(self, state):
        """Generate a simulated position reading (like from Lighthouse V2)."""
        # Extract true position
        true_position = state[0:3]
        
        # Add noise
        noise = np.random.normal(0, self.noise_levels['position'], true_position.shape)
        position_reading = true_position + noise
        
        return position_reading
    
    def generate_dataset(self, output_dir, num_trajectories, trajectory_duration, splits=None):
        """
        Generate a complete dataset of UAV trajectories and sensor readings.
        
        Args:
            output_dir: Directory to save the dataset
            num_trajectories: Number of trajectories to generate
            trajectory_duration: Duration of each trajectory in seconds
            splits: Dictionary of dataset splits (e.g. {'train': 0.7, 'val': 0.2, 'test': 0.1})
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Default splits if not provided
        if splits is None:
            splits = {'train': 0.7, 'val': 0.2, 'test': 0.1}
        
        # Create split directories
        for split in splits:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        
        # Calculate number of trajectories per split
        trajectories_per_split = {}
        remaining = num_trajectories
        for split, fraction in splits.items():
            if split == list(splits.keys())[-1]:
                # Last split gets the remainder
                trajectories_per_split[split] = remaining
            else:
                trajectories_per_split[split] = int(num_trajectories * fraction)
                remaining -= trajectories_per_split[split]
        
        # Generate trajectories and sensor data
        total_samples = 0
        
        for split, num_split_trajectories in trajectories_per_split.items():
            split_dir = os.path.join(output_dir, split)
            
            # Generate each trajectory
            for traj_idx in tqdm(range(num_split_trajectories), desc=f"Generating {split} trajectories"):
                # Generate trajectory
                timestamps, states = self.generate_trajectory(trajectory_duration)
                
                # Generate sensor data
                sensor_data = self.generate_sensor_data(states, timestamps)
                
                # Save trajectory and sensor data
                num_steps = len(timestamps)
                for step_idx in range(num_steps):
                    # Create sample ID
                    sample_id = f"{total_samples + step_idx:08d}"
                    
                    # Save state
                    state_path = os.path.join(split_dir, f"state_{sample_id}.npy")
                    np.save(state_path, states[step_idx])
                    
                    # Save IMU reading
                    imu_path = os.path.join(split_dir, f"imu_{sample_id}.npy")
                    np.save(imu_path, sensor_data['imu'][step_idx])
                    
                    # Save camera image
                    camera_path = os.path.join(split_dir, f"camera_{sample_id}.png")
                    cv2.imwrite(camera_path, cv2.cvtColor(sensor_data['camera'][step_idx], cv2.COLOR_RGB2BGR))
                    
                    # Save depth map
                    depth_path = os.path.join(split_dir, f"depth_{sample_id}.npy")
                    np.save(depth_path, sensor_data['depth'][step_idx])
                    
                    # Save position reading
                    position_path = os.path.join(split_dir, f"position_{sample_id}.npy")
                    np.save(position_path, sensor_data['position'][step_idx])
                
                total_samples += num_steps
        
        print(f"Generated {total_samples} samples across {num_trajectories} trajectories")

def main(args):
    # Initialize simulator
    simulator = UAVSimulator(
        dt=args.dt,
        trajectory_type=args.trajectory_type,
        environment_size=args.environment_size,
        camera_resolution=args.camera_resolution,
        depth_resolution=args.depth_resolution,
        max_velocity=args.max_velocity,
        max_angular_velocity=args.max_angular_velocity,
        noise_levels=args.noise_levels
    )
    
    # Generate dataset
    simulator.generate_dataset(
        output_dir=args.output_dir,
        num_trajectories=args.num_trajectories,
        trajectory_duration=args.trajectory_duration,
        splits=args.splits
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulated UAV dataset")
    
    # Simulation parameters
    parser.add_argument('--dt', type=float, default=0.01, help='Time step in seconds')
    parser.add_argument('--trajectory_type', type=str, default='random', 
                        choices=['random', 'circle', 'figure8', 'hover'],
                        help='Type of trajectory to generate')
    parser.add_argument('--environment_size', type=float, nargs=3, default=[10, 10, 5],
                        help='Size of the environment (x, y, z)')
    parser.add_argument('--camera_resolution', type=int, nargs=2, default=[64, 64],
                        help='Resolution of camera images (height, width)')
    parser.add_argument('--depth_resolution', type=int, nargs=2, default=[64, 64],
                        help='Resolution of depth maps (height, width)')
    parser.add_argument('--max_velocity', type=float, default=5.0,
                        help='Maximum linear velocity')
    parser.add_argument('--max_angular_velocity', type=float, default=3.0,
                        help='Maximum angular velocity')
    
    # Noise parameters
    parser.add_argument('--noise_levels', type=dict, default={
                            'imu': 0.02, 
                            'camera': 0.05, 
                            'depth': 0.1, 
                            'position': 0.05
                        }, help='Noise levels for different sensors')
    
    # Dataset parameters
    parser.add_argument('--output_dir', type=str, default='data/simulated',
                        help='Output directory for dataset')
    parser.add_argument('--num_trajectories', type=int, default=100,
                        help='Number of trajectories to generate')
    parser.add_argument('--trajectory_duration', type=float, default=10.0,
                        help='Duration of each trajectory in seconds')
    parser.add_argument('--splits', type=dict, default={'train': 0.7, 'val': 0.2, 'test': 0.1},
                        help='Dataset splits')
    
    args = parser.parse_args()
    
    # Import torch (needed for UAVPhysicsModel)
    import torch
    
    main(args) 