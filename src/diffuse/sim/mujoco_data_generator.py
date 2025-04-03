import os
import numpy as np
import argparse
import cv2
from tqdm import tqdm
import random
import time
import mujoco
import gymnasium as gym
from pathlib import Path
from dm_control import mujoco as dm_mujoco
import matplotlib.pyplot as plt
from PIL import Image

class CrazyflieSimulator:
    """
    Simulates Crazyflie UAV flights using MuJoCo to generate training data for the diffusion model.
    This simulator provides high-fidelity physics and sensor data approximating real-world conditions.
    Supports both simple and complex simulation modes.
    """
    def __init__(self, args):
        """Initialize the simulator with the given parameters."""
        # Get model path from args or use default
        model_path = getattr(args, 'model_path', 'src/diffuse/sim/assets/racing_quad.xml')
        
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Store parameters
        self.max_velocity = args.max_velocity
        self.max_angular_velocity = getattr(args, 'max_angular_velocity', 8.0)
        
        # Set up noise levels
        self.noise_levels = {
            'imu': 0.02,
            'camera': 0.05,
            'depth': 0.1,
            'position': 0.05
        }
        
        self.dt = args.dt
        self.image_size = args.camera_resolution
        self.depth_size = args.depth_resolution
        self.simple_mode = getattr(args, 'simple_mode', False)
        self.randomize_environment = getattr(args, 'randomize_environment', True) and not self.simple_mode
        self.randomize_lighting = getattr(args, 'randomize_lighting', True) and not self.simple_mode
        self.randomize_textures = getattr(args, 'randomize_textures', True) and not self.simple_mode
        
        # Initialize renderer only in complex mode
        if not self.simple_mode:
            self.renderer = mujoco.Renderer(self.model)
        
        # Get indices for sensors
        self.sensor_indices = {
            'accelerometer': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'accelerometer'),
            'gyro': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'gyro'),
            'orientation': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'orientation'),
        }
        
        # Add camera and depth sensors only in complex mode
        if not self.simple_mode:
            self.sensor_indices.update({
                'rgb': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'rgb'),
                'depth': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'depth')
            })
            self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'rgbd')
        
        # If randomizing environments in complex mode, prepare for it
        if self.randomize_environment:
            self._setup_randomized_environment()
    
    def _setup_randomized_environment(self):
        """Set up randomized environments with obstacles, lighting, and textures."""
        # This would be expanded in a real implementation to create more complex environments
        pass
    
    def _randomize_sim_parameters(self):
        """Randomize simulation parameters for domain randomization."""
        if self.randomize_lighting:
            # Randomize light position and intensity
            for i in range(self.model.nlight):
                # Randomize light position
                self.model.light_pos[i, 0] += random.uniform(-1.0, 1.0)
                self.model.light_pos[i, 1] += random.uniform(-1.0, 1.0)
                self.model.light_pos[i, 2] += random.uniform(0.5, 2.0)
                
                # Randomize light intensity (diffuse)
                self.model.light_diffuse[i, 0] = random.uniform(0.5, 1.0)
                self.model.light_diffuse[i, 1] = random.uniform(0.5, 1.0)
                self.model.light_diffuse[i, 2] = random.uniform(0.5, 1.0)
        
        if self.randomize_textures:
            # In a real implementation, this would modify texture parameters or load different textures
            pass
    
    def _apply_control(self, action):
        """Apply control inputs to the UAV."""
        # Convert normalized actions [-1, 1] to motor commands [0, 1]
        motor_commands = (action + 1) / 2.0
        
        # Apply motor commands
        self.data.ctrl[:] = motor_commands
    
    def _get_state(self):
        """Get the current UAV state (position, orientation, velocities)."""
        # Get body position and orientation (Euler angles)
        position = self.data.qpos[:3].copy()
        orientation = self.data.qpos[3:6].copy()  # roll, pitch, yaw
        
        # Get velocities
        linear_velocity = self.data.qvel[:3].copy()
        angular_velocity = self.data.qvel[3:6].copy()
        
        # Combine into state vector
        state = np.concatenate([position, orientation, linear_velocity, angular_velocity])
        return state
    
    def _get_sensor_data(self):
        """Get sensor readings from the simulation."""
        # Create dictionary to store sensor readings
        sensor_data = {}
        
        # Get IMU data (accelerometer + gyro)
        accel = self.data.sensor(self.sensor_indices['accelerometer']).data.copy()
        gyro = self.data.sensor(self.sensor_indices['gyro']).data.copy()
        
        # Add noise to IMU data
        accel_noise = np.random.normal(0, self.noise_levels['imu'], accel.shape)
        gyro_noise = np.random.normal(0, self.noise_levels['imu'], gyro.shape)
        
        accel += accel_noise
        gyro += gyro_noise
        
        # Combine into IMU reading
        sensor_data['imu'] = np.concatenate([accel, gyro])
        
        # Get position data (from state)
        position = self.data.qpos[:3].copy()
        
        # Add noise to position
        position_noise = np.random.normal(0, self.noise_levels['position'], position.shape)
        position += position_noise
        
        sensor_data['position'] = position
        
        if not self.simple_mode:
            # Get RGB camera image using renderer
            self.renderer.update_scene(self.data, camera=self.camera_id)
            rgb_img = self.renderer.render().copy()
            
            # Resize image to desired dimensions
            rgb_img = cv2.resize(rgb_img, (self.image_size[1], self.image_size[0]))
            
            # Add noise and perturbations to image
            camera_noise = np.random.normal(0, self.noise_levels['camera'], rgb_img.shape) * 255
            rgb_img = np.clip(rgb_img + camera_noise, 0, 255).astype(np.uint8)
            
            # Adjust brightness, contrast randomly
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            rgb_img = cv2.convertScaleAbs(rgb_img, alpha=contrast, beta=brightness * 5)
            
            sensor_data['camera'] = rgb_img
            
            # Get depth image from renderer
            depth_img = np.zeros(self.depth_size)  # Placeholder
            
            # Add noise to depth
            depth_noise = np.random.normal(0, self.noise_levels['depth'], depth_img.shape)
            depth_img += depth_noise
            
            sensor_data['depth'] = depth_img
        else:
            # Create simple placeholder images for camera and depth
            # Simple checkerboard pattern for camera
            camera = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            for i in range(8):
                for j in range(8):
                    if (i + j) % 2 == 0:
                        camera[i*8:(i+1)*8, j*8:(j+1)*8, :] = 200
            sensor_data['camera'] = camera
            
            # Zero-filled depth map
            sensor_data['depth'] = np.zeros(self.depth_size)
        
        return sensor_data
    
    def reset(self):
        """Reset the simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial position
        self.data.qpos[:3] = np.random.uniform(-1, 1, 3)
        self.data.qpos[2] = abs(self.data.qpos[2]) + 0.5  # Ensure positive z
        
        # Randomize initial orientation (Euler angles)
        if random.random() < 0.7:  # 70% of the time, start close to level
            self.data.qpos[3:6] = [0, 0, 0]  # Level orientation (roll, pitch, yaw)
        else:  # 30% with random orientation
            self.data.qpos[3:6] = np.random.uniform(-0.3, 0.3, 3)  # Small random angles
        
        # Optional: randomize environment
        if self.randomize_environment:
            self._randomize_sim_parameters()
        
        # Forward the simulation to update derived quantities
        mujoco.mj_forward(self.model, self.data)
        
        # Return initial state and sensor data
        return self._get_state(), self._get_sensor_data()
    
    def step(self, target_state):
        """Step the simulation towards the target state."""
        # Extract target state components
        target_pos = target_state['position']
        target_rot = target_state['orientation']
        target_vel = target_state['velocity']
        
        # Set target position and orientation
        self.data.qpos[:3] = target_pos
        self.data.qpos[3:6] = target_rot
        
        # Set target velocity
        self.data.qvel[:3] = target_vel
        self.data.qvel[3:6] = np.zeros(3)  # Angular velocity
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        
        # Return sensor data
        return self._get_sensor_data()
    
    def generate_trajectory(self, trajectory_type, duration):
        """
        Generate a UAV trajectory of specified type and duration.
        
        Args:
            trajectory_type: Type of trajectory ('random', 'circle', 'figure8', 'hover')
            duration: Duration of trajectory in seconds
            
        Returns:
            states: List of UAV states
            all_sensor_data: List of sensor readings
        """
        # Calculate number of steps
        n_steps = int(duration / self.dt)
        
        # Initialize trajectory data
        states = []
        all_sensor_data = []
        
        # Reset simulation
        state, sensor_data = self.reset()
        states.append(state)
        all_sensor_data.append(sensor_data)
        
        # Generate trajectory
        for i in range(1, n_steps):
            # Calculate control action based on trajectory type
            if trajectory_type == 'random':
                # Random control actions
                action = np.random.uniform(-0.5, 0.5, 4)
                # Add base thrust to counteract gravity (around 0.5-0.7)
                action += 0.6
                
            elif trajectory_type == 'hover':
                # Simple hover at current position
                action = np.array([0.5, 0.5, 0.5, 0.5])
                
            elif trajectory_type == 'circle':
                # Circle trajectory
                t = i * self.dt
                freq = 0.5  # Hz
                
                # Calculate desired yaw angle for circle
                target_yaw = 2 * np.pi * freq * t
                
                # Simple controller to follow circular path
                current_position = states[-1][:3]
                current_yaw = states[-1][5]  # Assuming state[5] is yaw
                
                # Basic control law to move in a circle
                # Adjust thrust slightly based on altitude error
                base_thrust = 0.6
                altitude_error = 1.0 - current_position[2]
                thrust_adjustment = 0.1 * altitude_error
                
                # Differential thrust for yaw control
                yaw_error = target_yaw - current_yaw
                yaw_adjustment = 0.05 * yaw_error
                
                # Simple control allocation (this would be more sophisticated in reality)
                action = np.array([
                    base_thrust + thrust_adjustment + yaw_adjustment,
                    base_thrust + thrust_adjustment - yaw_adjustment,
                    base_thrust + thrust_adjustment + yaw_adjustment,
                    base_thrust + thrust_adjustment - yaw_adjustment
                ])
                
            elif trajectory_type == 'figure8':
                # Figure-8 trajectory
                t = i * self.dt
                freq = 0.3  # Hz
                
                # Calculate desired position on figure-8
                phase_x = 2 * np.pi * freq * t
                phase_y = 2 * np.pi * freq * t * 2
                
                target_x = np.sin(phase_x)
                target_y = np.sin(phase_y)
                target_z = 1.0  # Fixed altitude
                
                # Current position
                current_position = states[-1][:3]
                
                # Simple control allocation (would be more sophisticated in practice)
                position_error = np.array([
                    target_x - current_position[0],
                    target_y - current_position[1],
                    target_z - current_position[2]
                ])
                
                # Basic PD control-inspired approach
                base_thrust = 0.6
                x_adjustment = 0.1 * position_error[0]
                y_adjustment = 0.1 * position_error[1]
                z_adjustment = 0.1 * position_error[2]
                
                # Simple thrust mixing (this is a vast simplification)
                action = np.array([
                    base_thrust + z_adjustment - y_adjustment + x_adjustment,
                    base_thrust + z_adjustment - y_adjustment - x_adjustment,
                    base_thrust + z_adjustment + y_adjustment - x_adjustment,
                    base_thrust + z_adjustment + y_adjustment + x_adjustment
                ])
            
            # Clip actions to valid range
            action = np.clip(action, 0, 1)
            
            # Step the simulation
            state, sensor_data = self.step(action)
            
            # Store data
            states.append(state)
            all_sensor_data.append(sensor_data)
        
        return states, all_sensor_data

    def simulate_trajectory(self, positions, rotations):
        """Simulate a trajectory with given positions and rotations.
        
        Args:
            positions: Array of target positions [N, 3]
            rotations: Array of target rotations (euler angles) [N, 3]
            
        Returns:
            Dictionary of sensor data for the trajectory
        """
        # Reset simulation
        state, sensor_data = self.reset()
        
        # Initialize data storage
        all_sensor_data = {
            'imu': [],
            'position': [],
            'camera': None,  # We'll only save the first frame for memory efficiency
            'fpv': None
        }
        
        # Simple PD controller gains
        Kp_pos = 5.0
        Kd_pos = 2.0
        Kp_att = 3.0
        Kd_att = 1.0
        
        # Get initial state
        current_pos = self.data.qpos[:3]
        current_vel = self.data.qvel[:3]
        current_rot = self.data.qpos[3:6]  # Assuming Euler angles
        current_ang_vel = self.data.qvel[3:6]
        
        # Simulate trajectory
        for target_pos, target_rot in zip(positions, rotations):
            # Compute control errors
            pos_error = target_pos - current_pos
            vel_error = -current_vel  # Try to minimize velocity
            rot_error = target_rot - current_rot
            ang_vel_error = -current_ang_vel
            
            # PD control law
            force = Kp_pos * pos_error + Kd_pos * vel_error
            torque = Kp_att * rot_error + Kd_att * ang_vel_error
            
            # Convert to motor commands (simplified allocation)
            thrust = force[2] + 9.81  # Add gravity compensation
            roll_cmd = torque[0]
            pitch_cmd = torque[1]
            yaw_cmd = torque[2]
            
            # Motor mixing (simplified)
            m1 = thrust + roll_cmd + pitch_cmd + yaw_cmd
            m2 = thrust - roll_cmd + pitch_cmd - yaw_cmd
            m3 = thrust - roll_cmd - pitch_cmd + yaw_cmd
            m4 = thrust + roll_cmd - pitch_cmd - yaw_cmd
            
            # Normalize and clip motor commands
            motors = np.array([m1, m2, m3, m4])
            motors = np.clip(motors / motors.max(), 0, 1)
            
            # Apply control
            self.data.ctrl[:] = motors
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Get sensor readings
            sensors = self._get_sensor_data()
            
            # Store sensor data
            all_sensor_data['imu'].append(sensors['imu'])
            all_sensor_data['position'].append(sensors['position'])
            
            # Store only first camera frame
            if all_sensor_data['camera'] is None and 'camera' in sensors:
                all_sensor_data['camera'] = sensors['camera']
            if all_sensor_data['fpv'] is None and 'fpv' in sensors:
                all_sensor_data['fpv'] = sensors['camera']  # Using main camera for FPV
            
            # Update current state
            current_pos = self.data.qpos[:3]
            current_vel = self.data.qvel[:3]
            current_rot = self.data.qpos[3:6]
            current_ang_vel = self.data.qvel[3:6]
        
        # Convert lists to arrays
        all_sensor_data['imu'] = np.array(all_sensor_data['imu'])
        all_sensor_data['position'] = np.array(all_sensor_data['position'])
        
        return all_sensor_data

def save_trajectory_data(output_dir, traj_idx, states, sensor_data, split='train'):
    """
    Save trajectory data to disk in the format expected by the dataset loader.
    
    Args:
        output_dir: Base output directory
        traj_idx: Trajectory index
        states: List of UAV states
        sensor_data: List of sensor readings
        split: Dataset split ('train', 'val', 'test')
    """
    # Create split directory if it doesn't exist
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # Save each time step
    for t, (state, sensors) in enumerate(zip(states, sensor_data)):
        # Generate unique sample ID
        sample_id = f"{traj_idx:04d}_{t:04d}"
        
        # Save clean state
        state_file = os.path.join(split_dir, f"state_{sample_id}.npy")
        np.save(state_file, state)
        
        # Save IMU data
        if 'imu' in sensors:
            imu_file = os.path.join(split_dir, f"imu_{sample_id}.npy")
            np.save(imu_file, sensors['imu'])
        
        # Save position data
        if 'position' in sensors:
            position_file = os.path.join(split_dir, f"position_{sample_id}.npy")
            np.save(position_file, sensors['position'])
        
        # Save camera image
        if 'camera' in sensors:
            camera_file = os.path.join(split_dir, f"camera_{sample_id}.png")
            cv2.imwrite(camera_file, cv2.cvtColor(sensors['camera'], cv2.COLOR_RGB2BGR))
        
        # Save depth map
        if 'depth' in sensors:
            depth_file = os.path.join(split_dir, f"depth_{sample_id}.npy")
            np.save(depth_file, sensors['depth'])

def generate_racing_trajectory(t, args):
    """Generate an aggressive racing trajectory."""
    frequency = args.trajectory_frequency
    aggressiveness = args.racing_aggressiveness
    min_height = args.min_height
    max_velocity = args.max_velocity
    
    # Base figure-8 pattern with varying height
    x = aggressiveness * np.sin(2 * np.pi * frequency * t)
    y = aggressiveness * np.sin(2 * np.pi * frequency * t * 2)
    z = min_height + 0.5 * (1 + np.sin(2 * np.pi * frequency * t * 0.5))
    
    # Add some variation to make it more dynamic
    x += 0.3 * np.sin(2 * np.pi * frequency * t * 3)
    y += 0.3 * np.cos(2 * np.pi * frequency * t * 2.5)
    
    # Calculate velocities
    dx = 2 * np.pi * frequency * aggressiveness * np.cos(2 * np.pi * frequency * t)
    dy = 4 * np.pi * frequency * aggressiveness * np.cos(4 * np.pi * frequency * t)
    dz = np.pi * frequency * np.cos(np.pi * frequency * t)
    
    # Normalize velocity to respect max_velocity
    velocity = np.array([dx, dy, dz])
    speed = np.linalg.norm(velocity)
    if speed > max_velocity:
        velocity = velocity * (max_velocity / speed)
    
    # Calculate desired orientation (align with velocity direction)
    yaw = np.arctan2(velocity[1], velocity[0])
    pitch = -np.arctan2(velocity[2], np.sqrt(velocity[0]**2 + velocity[1]**2))
    roll = 0.2 * np.sin(2 * np.pi * frequency * t * 2)  # Add some roll variation
    
    # Return position and orientation
    position = np.array([x, y, z])
    orientation = np.array([roll, pitch, yaw])
    
    return {'position': position, 'orientation': orientation, 'velocity': velocity}

def generate_trajectory(t, args):
    """Generate a simple trajectory based on the specified type."""
    if args.trajectory_type == 'circle':
        radius = 1.0
        frequency = args.trajectory_frequency
        x = radius * np.cos(2 * np.pi * frequency * t)
        y = radius * np.sin(2 * np.pi * frequency * t)
        z = args.min_height
        yaw = 2 * np.pi * frequency * t
        
        position = np.array([x, y, z])
        orientation = np.array([0, 0, yaw])
        velocity = np.array([-2 * np.pi * frequency * y, 2 * np.pi * frequency * x, 0])
        
    elif args.trajectory_type == 'hover':
        position = np.array([0, 0, args.min_height])
        orientation = np.array([0, 0, 0])
        velocity = np.zeros(3)
        
    else:  # random trajectory
        t_scaled = t * args.trajectory_frequency
        position = np.array([
            np.sin(2 * np.pi * t_scaled),
            np.cos(2 * np.pi * t_scaled * 0.5),
            args.min_height + 0.5 * (1 + np.sin(2 * np.pi * t_scaled * 0.3))
        ])
        orientation = np.array([
            0.1 * np.sin(2 * np.pi * t_scaled * 2),
            0.1 * np.cos(2 * np.pi * t_scaled),
            2 * np.pi * t_scaled
        ])
        velocity = np.array([
            2 * np.pi * args.trajectory_frequency * np.cos(2 * np.pi * t_scaled),
            -np.pi * args.trajectory_frequency * np.sin(np.pi * t_scaled),
            0.3 * np.pi * args.trajectory_frequency * np.cos(0.6 * np.pi * t_scaled)
        ])
    
    # Scale velocity to respect max_velocity
    speed = np.linalg.norm(velocity)
    if speed > args.max_velocity:
        velocity = velocity * (args.max_velocity / speed)
    
    return {'position': position, 'orientation': orientation, 'velocity': velocity}

def generate_dataset(args):
    """Generate dataset using MuJoCo simulation."""
    np.random.seed(args.seed)
    
    # Initialize simulator
    simulator = CrazyflieSimulator(args)
    
    # Calculate number of timesteps
    timesteps = int(args.trajectory_duration / args.dt)
    
    # Calculate split sizes
    n_train = int(args.num_trajectories * args.splits[0])
    n_val = int(args.num_trajectories * args.splits[1])
    n_test = args.num_trajectories - n_train - n_val
    
    # Generate trajectories for each split
    splits = [
        ('train', n_train),
        ('val', n_val),
        ('test', n_test)
    ]
    
    for split_name, n_trajectories in splits:
        print(f"\nGenerating {n_trajectories} trajectories for {split_name} set...")
        
        for i in range(n_trajectories):
            # Generate trajectory
            t = np.linspace(0, args.trajectory_duration, timesteps)
            positions = []
            rotations = []
            
            for tj in t:
                pos, rot = generate_trajectory(tj, args)
                positions.append(pos)
                rotations.append(rot)
            
            # Convert to arrays
            positions = np.array(positions)
            rotations = np.array(rotations)
            
            # Simulate trajectory and collect data
            data = simulator.simulate_trajectory(positions, rotations)
            
            # Save data
            save_path = os.path.join(args.output_dir, split_name)
            os.makedirs(save_path, exist_ok=True)
            
            # Save each sensor reading
            for sensor_name, sensor_data in data.items():
                if sensor_data is None:  # Skip if no data
                    continue
                
                filename = f"{sensor_name}_{i:04d}_{0:04d}"
                if sensor_name in ['camera', 'fpv']:
                    # Create a dummy image if camera data is not available
                    if sensor_data is None:
                        sensor_data = np.zeros((64, 64, 3), dtype=np.uint8)
                    cv2.imwrite(os.path.join(save_path, f"{filename}.png"), sensor_data)
                else:
                    np.save(os.path.join(save_path, f"{filename}.npy"), sensor_data)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{n_trajectories} trajectories")
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MuJoCo-based Crazyflie UAV dataset")
    
    # Simulation parameters
    parser.add_argument('--model_path', type=str, default='src/diffuse/sim/assets/crazyflie.xml',
                        help='Path to MuJoCo XML model file')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step in seconds')
    parser.add_argument('--trajectory_type', type=str, default='mixed',
                        choices=['random', 'circle', 'figure8', 'hover', 'mixed'],
                        help='Type of trajectory to generate')
    parser.add_argument('--camera_resolution', type=int, nargs=2, default=[64, 64],
                        help='Resolution of camera images (height, width)')
    parser.add_argument('--depth_resolution', type=int, nargs=2, default=[64, 64],
                        help='Resolution of depth maps (height, width)')
    parser.add_argument('--max_velocity', type=float, default=5.0,
                        help='Maximum linear velocity')
    parser.add_argument('--max_angular_velocity', type=float, default=3.0,
                        help='Maximum angular velocity')
    parser.add_argument('--noise_levels', type=float, nargs=4, default=[0.02, 0.05, 0.1, 0.05],
                        help='Noise levels for IMU, camera, depth, position')
    
    # Domain randomization options
    parser.add_argument('--randomize_environment', action='store_true',
                        help='Randomize environment geometry')
    parser.add_argument('--randomize_lighting', action='store_true',
                        help='Randomize lighting conditions')
    parser.add_argument('--randomize_textures', action='store_true',
                        help='Randomize textures')
    
    # Dataset parameters
    parser.add_argument('--output_dir', type=str, default='data/simulated',
                        help='Output directory for dataset')
    parser.add_argument('--num_trajectories', type=int, default=100,
                        help='Number of trajectories to generate')
    parser.add_argument('--trajectory_duration', type=float, default=5.0,
                        help='Duration of each trajectory in seconds')
    parser.add_argument('--splits', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Dataset splits (train, val, test)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Convert noise levels to dictionary
    args.noise_levels = {
        'imu': args.noise_levels[0],
        'camera': args.noise_levels[1],
        'depth': args.noise_levels[2],
        'position': args.noise_levels[3]
    }
    
    generate_dataset(args) 