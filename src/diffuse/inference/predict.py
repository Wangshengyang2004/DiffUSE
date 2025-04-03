import torch
import numpy as np
import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from diffuse.models.diffusion_model import DiffusionStateEstimator
from diffuse.models.diffusion_process import DiffusionProcess
from diffuse.models.physics_model import UAVPhysicsModel

def load_model(model_path, state_dim=12, sensor_channels=None, device='cpu'):
    """
    Load a pretrained diffusion model.
    
    Args:
        model_path: Path to the model checkpoint
        state_dim: Dimension of UAV state
        sensor_channels: Dictionary of sensor channel counts
        device: Device to load the model on
        
    Returns:
        model: Loaded model
        diffusion: Diffusion process
    """
    # Default sensor channels if not provided
    if sensor_channels is None:
        sensor_channels = {
            'imu': 6,
            'camera': 3,
            'depth': 1,
            'position': 3
        }
    
    # Initialize model
    model = DiffusionStateEstimator(
        state_dim=state_dim,
        sensor_channels=sensor_channels
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(model=model)
    
    return model, diffusion

def prepare_sensor_data(sensor_readings, device='cpu'):
    """
    Prepare sensor data for model input.
    
    Args:
        sensor_readings: Dictionary of raw sensor readings
        device: Device to move tensors to
        
    Returns:
        sensor_data: Dictionary of processed sensor data as tensors
    """
    sensor_data = {}
    
    for sensor_type, data in sensor_readings.items():
        if sensor_type == 'camera':
            # Process image
            if isinstance(data, np.ndarray):
                # Normalize to [0, 1] and convert to CHW format
                if data.dtype == np.uint8:
                    data = data.astype(np.float32) / 255.0
                
                if len(data.shape) == 3 and data.shape[2] == 3:  # HWC format
                    data = data.transpose(2, 0, 1)  # Convert to CHW
                
                data = torch.from_numpy(data).float()
            
            # Ensure batch dimension
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
                
        elif sensor_type == 'depth':
            # Process depth map
            if isinstance(data, np.ndarray):
                # Normalize depth
                if data.max() > 1.0:
                    data = data / data.max()
                    
                # Add channel dimension if needed
                if len(data.shape) == 2:
                    data = data[np.newaxis, :, :]
                
                data = torch.from_numpy(data).float()
            
            # Ensure batch dimension
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
                
        elif sensor_type in ['imu', 'position']:
            # Process vector data
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data.astype(np.float32))
            
            # Ensure batch dimension
            if len(data.shape) == 1:
                data = data.unsqueeze(0)
        
        # Move to device
        sensor_data[sensor_type] = data.to(device)
    
    return sensor_data

def estimate_state(diffusion, sensor_data, initial_state=None, use_physics=False, 
                   device='cpu', diffusion_steps=1000, guide_scale=0.0, 
                   verbose=True):
    """
    Estimate UAV state using diffusion model.
    
    Args:
        diffusion: Diffusion process
        sensor_data: Dictionary of sensor readings
        initial_state: Optional initial state estimate
        use_physics: Whether to use physics constraints
        device: Device to run inference on
        diffusion_steps: Number of diffusion steps to use
        guide_scale: Guidance scale for classifier-free guidance (0.0 = disabled)
        verbose: Whether to show progress
        
    Returns:
        estimated_state: Estimated clean state
    """
    # Initialize physics model if needed
    physics_model = UAVPhysicsModel() if use_physics else None
    
    # Use provided initial state or create one
    if initial_state is None:
        # Get batch size from first sensor data
        batch_size = next(iter(sensor_data.values())).shape[0]
        initial_state = torch.zeros((batch_size, diffusion.model.state_dim), device=device)
    
    # Adjust diffusion steps (can use fewer steps for faster inference)
    original_steps = diffusion.n_timesteps
    diffusion.n_timesteps = diffusion_steps
    
    start_time = time.time()
    
    # Use physical corrected sampling if requested
    if use_physics:
        estimated_state = diffusion.physical_corrected_sample(
            initial_state=initial_state,
            sensor_data=sensor_data,
            physics_model=physics_model,
            device=device,
            show_progress=verbose
        )
    else:
        estimated_state = diffusion.sample(
            initial_state=initial_state,
            sensor_data=sensor_data,
            device=device,
            show_progress=verbose
        )
    
    inference_time = time.time() - start_time
    
    # Restore original steps
    diffusion.n_timesteps = original_steps
    
    if verbose:
        print(f"Inference completed in {inference_time:.2f}s")
    
    return estimated_state

def main(args):
    # Set device
    device = torch.device(args.device)
    
    # Load model
    model, diffusion = load_model(
        model_path=args.model_path,
        device=device
    )
    
    # Load sensor data
    sensor_readings = {}
    
    # These paths would be replaced with actual sensor reading code in a real UAV system
    if args.imu_file:
        sensor_readings['imu'] = np.load(args.imu_file)
    
    if args.camera_file:
        import cv2
        camera = cv2.imread(args.camera_file)
        camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        sensor_readings['camera'] = camera
    
    if args.depth_file:
        sensor_readings['depth'] = np.load(args.depth_file)
    
    if args.position_file:
        sensor_readings['position'] = np.load(args.position_file)
    
    # Prepare sensor data
    sensor_data = prepare_sensor_data(sensor_readings, device)
    
    # Load initial state if provided
    initial_state = None
    if args.initial_state_file:
        initial_state = np.load(args.initial_state_file)
        initial_state = torch.from_numpy(initial_state).float().to(device)
        
        # Add batch dimension if needed
        if len(initial_state.shape) == 1:
            initial_state = initial_state.unsqueeze(0)
    
    # Estimate state
    estimated_state = estimate_state(
        diffusion=diffusion,
        sensor_data=sensor_data,
        initial_state=initial_state,
        use_physics=args.use_physics,
        device=device,
        diffusion_steps=args.diffusion_steps,
        verbose=True
    )
    
    # Convert to numpy
    estimated_state = estimated_state.cpu().numpy()
    
    # Save result if output path provided
    if args.output_file:
        np.save(args.output_file, estimated_state)
        print(f"Estimated state saved to {args.output_file}")
    
    # Print result
    print("Estimated state:")
    print(f"Position: {estimated_state[0, 0:3]}")
    print(f"Orientation: {estimated_state[0, 3:6]}")
    print(f"Linear Velocity: {estimated_state[0, 6:9]}")
    print(f"Angular Velocity: {estimated_state[0, 9:12]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV state estimation using diffusion model")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    
    # Sensor data
    parser.add_argument('--imu_file', type=str, help='Path to IMU data file (.npy)')
    parser.add_argument('--camera_file', type=str, help='Path to camera image file')
    parser.add_argument('--depth_file', type=str, help='Path to depth map file (.npy)')
    parser.add_argument('--position_file', type=str, help='Path to position data file (.npy)')
    
    # Initial state
    parser.add_argument('--initial_state_file', type=str, help='Path to initial state file (.npy)')
    
    # Inference parameters
    parser.add_argument('--use_physics', action='store_true', help='Use physics model for constraints')
    parser.add_argument('--diffusion_steps', type=int, default=50, help='Number of diffusion steps for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    
    # Output
    parser.add_argument('--output_file', type=str, help='Path to save estimated state (.npy)')
    
    args = parser.parse_args()
    main(args) 