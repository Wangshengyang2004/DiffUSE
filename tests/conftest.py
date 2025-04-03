"""
Pytest configuration and fixtures.
"""
import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add src directory to path for imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.diffuse.models.diffusion_model import DiffusionStateEstimator
from src.diffuse.models.diffusion_process import DiffusionProcess
from src.diffuse.models.physics_model import UAVPhysicsModel


@pytest.fixture(scope="session")
def device():
    """Get PyTorch device (CPU or GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@pytest.fixture
def sample_uav_state():
    """Generate a sample UAV state vector."""
    # [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    # position: [px, py, pz]
    # quaternion: [qw, qx, qy, qz]
    # linear velocity: [vx, vy, vz]
    # angular velocity: [wx, wy, wz]
    
    position = torch.tensor([1.0, 2.0, 3.0])
    
    # Unit quaternion for orientation (normalized)
    quat = torch.tensor([0.707, 0.0, 0.707, 0.0])
    quat = quat / torch.norm(quat)
    
    lin_vel = torch.tensor([0.5, 0.0, -0.2])
    ang_vel = torch.tensor([0.1, 0.2, -0.1])
    
    state = torch.cat([position, quat, lin_vel, ang_vel])
    return state


@pytest.fixture
def sample_batch(sample_uav_state):
    """Generate a sample batch for training."""
    batch_size = 8
    
    # Create batch of clean states
    clean_states = torch.stack([sample_uav_state + torch.randn_like(sample_uav_state) * 0.1 
                               for _ in range(batch_size)])
    
    # Create sensor data
    imu_data = torch.randn(batch_size, 6)  # 3D accel + 3D gyro
    camera_data = torch.randn(batch_size, 3, 64, 64)  # RGB image
    depth_data = torch.randn(batch_size, 1, 64, 64)  # Depth map
    position_data = torch.randn(batch_size, 3)  # Noisy position
    
    # Combined sensor data
    sensor_embedding = torch.randn(batch_size, 64)  # Pretend this is the embedding
    
    return {
        'clean_state': clean_states,
        'sensor_data': {
            'imu': imu_data,
            'camera': camera_data,
            'depth': depth_data,
            'position': position_data,
            'combined': sensor_embedding
        }
    }


@pytest.fixture
def setup_gpu():
    """Setup GPU if available for tests."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return {'device': device}


@pytest.fixture
def sample_sensor_data(batch_size=2):
    """Generate sample sensor data for testing."""
    # Define sample images and sensor readings
    camera = torch.rand(batch_size, 3, 64, 64)  # RGB image [B, C, H, W]
    depth = torch.rand(batch_size, 1, 64, 64)   # Depth map [B, C, H, W]
    imu = torch.rand(batch_size, 6)             # IMU readings [B, 6]
    position = torch.rand(batch_size, 3)        # Position data [B, 3]
    
    return {
        'camera': camera,
        'depth': depth,
        'imu': imu,
        'position': position
    }


@pytest.fixture
def diffusion_model(device):
    """Create a small diffusion model for testing."""
    model = DiffusionStateEstimator(
        state_dim=12,
        time_embedding_dim=32,
        sensor_channels={'imu': 6, 'camera': 3, 'depth': 1, 'position': 3},
        sensor_embedding_dim=32,
        hidden_dims=[32, 64, 32],
        use_attention=False
    ).to(device)
    
    return model


@pytest.fixture
def diffusion_process(diffusion_model):
    """Create a diffusion process for testing."""
    diffusion = DiffusionProcess(
        model=diffusion_model,
        n_timesteps=10,  # Use a small number of timesteps for faster testing
        beta_schedule='linear',
        beta_start=1e-4,
        beta_end=0.02
    )
    
    return diffusion


@pytest.fixture
def physics_model():
    """Create a physics model for testing."""
    return UAVPhysicsModel(dt=0.01) 