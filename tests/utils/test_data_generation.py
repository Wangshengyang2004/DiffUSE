"""
Tests for data generation utilities.
"""
import pytest
import torch
import numpy as np
import os
from pathlib import Path
import shutil

from src.diffuse.utils.data_generation import (
    SimulationConfig, 
    UAVSimulator,
    generate_simulation_data
)


class TestSimulationConfig:
    
    def test_init(self):
        """Test initialization of SimulationConfig."""
        config = SimulationConfig(
            num_trajectories=10,
            trajectory_length=100,
            dt=0.02,
            add_sensor_noise=True,
            camera_noise_std=0.05,
            imu_noise_std=0.02,
            position_noise_std=0.1,
            depth_noise_std=0.03,
            domain_randomization=True,
            seed=42
        )
        
        assert config.num_trajectories == 10
        assert config.trajectory_length == 100
        assert config.dt == 0.02
        assert config.add_sensor_noise
        assert config.camera_noise_std == 0.05
        assert config.imu_noise_std == 0.02
        assert config.position_noise_std == 0.1
        assert config.depth_noise_std == 0.03
        assert config.domain_randomization
        assert config.seed == 42
        
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'num_trajectories': 10,
            'trajectory_length': 100,
            'dt': 0.02,
            'add_sensor_noise': True,
            'camera_noise_std': 0.05,
            'imu_noise_std': 0.02,
            'position_noise_std': 0.1,
            'depth_noise_std': 0.03,
            'domain_randomization': True,
            'seed': 42
        }
        
        config = SimulationConfig.from_dict(config_dict)
        
        assert config.num_trajectories == 10
        assert config.trajectory_length == 100
        assert config.dt == 0.02
        assert config.add_sensor_noise
        
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = SimulationConfig(
            num_trajectories=10,
            trajectory_length=100,
            dt=0.02,
            add_sensor_noise=True,
            camera_noise_std=0.05
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['num_trajectories'] == 10
        assert config_dict['trajectory_length'] == 100
        assert config_dict['dt'] == 0.02
        assert config_dict['add_sensor_noise'] is True
        assert config_dict['camera_noise_std'] == 0.05
        
    def test_save_and_load(self, tmp_path):
        """Test saving and loading config."""
        config = SimulationConfig(
            num_trajectories=10,
            trajectory_length=100,
            dt=0.02
        )
        
        # Save config
        config_path = tmp_path / 'config.json'
        config.save(config_path)
        
        # Check file exists
        assert os.path.exists(config_path)
        
        # Load config
        loaded_config = SimulationConfig.load(config_path)
        
        # Check values
        assert loaded_config.num_trajectories == 10
        assert loaded_config.trajectory_length == 100
        assert loaded_config.dt == 0.02


class TestUAVSimulator:
    
    @pytest.fixture
    def simulator(self):
        """Create a UAV simulator for testing."""
        return UAVSimulator(dt=0.02)
    
    def test_init(self, simulator):
        """Test initialization of UAVSimulator."""
        assert simulator.dt == 0.02
        assert simulator.physics_model is not None
        
    def test_reset(self, simulator):
        """Test resetting the simulator."""
        # Reset with default parameters
        state = simulator.reset()
        
        # Check state shape
        assert state.shape == (12,)
        
        # Reset with custom position
        custom_pos = torch.tensor([1.0, 2.0, 3.0])
        state = simulator.reset(position=custom_pos)
        
        # Check position
        assert torch.allclose(state[:3], custom_pos)
        
        # Reset with custom orientation
        custom_quat = torch.tensor([0.707, 0.0, 0.707, 0.0])
        custom_quat = custom_quat / torch.norm(custom_quat)
        state = simulator.reset(orientation=custom_quat)
        
        # Check orientation
        assert torch.allclose(state[3:7], custom_quat)
        
    def test_step(self, simulator):
        """Test stepping the simulator."""
        # Reset simulator
        state = simulator.reset()
        
        # Create random action
        action = torch.randn(4)  # 4D control input for UAV
        
        # Step simulator
        next_state, sensor_data = simulator.step(action)
        
        # Check state shape
        assert next_state.shape == (12,)
        
        # Check sensor data
        assert 'imu' in sensor_data
        assert 'camera' in sensor_data
        assert 'depth' in sensor_data
        assert 'position' in sensor_data
        
        assert sensor_data['imu'].shape == (6,)  # 3D accel + 3D gyro
        assert sensor_data['camera'].shape == (3, 64, 64)  # [C, H, W]
        assert sensor_data['depth'].shape == (1, 64, 64)  # [C, H, W]
        assert sensor_data['position'].shape == (3,)
        
    def test_add_sensor_noise(self, simulator):
        """Test adding noise to sensor readings."""
        # Reset simulator
        state = simulator.reset()
        
        # Create clean sensor readings
        clean_imu = torch.zeros(6)
        clean_camera = torch.zeros(3, 64, 64)
        clean_depth = torch.zeros(1, 64, 64)
        clean_position = torch.zeros(3)
        
        # Add noise
        noisy_imu = simulator.add_imu_noise(clean_imu, std=0.1)
        noisy_camera = simulator.add_camera_noise(clean_camera, std=0.1)
        noisy_depth = simulator.add_depth_noise(clean_depth, std=0.1)
        noisy_position = simulator.add_position_noise(clean_position, std=0.1)
        
        # Check shapes
        assert noisy_imu.shape == clean_imu.shape
        assert noisy_camera.shape == clean_camera.shape
        assert noisy_depth.shape == clean_depth.shape
        assert noisy_position.shape == clean_position.shape
        
        # Check noise was added
        assert torch.norm(noisy_imu - clean_imu) > 0
        assert torch.norm(noisy_camera - clean_camera) > 0
        assert torch.norm(noisy_depth - clean_depth) > 0
        assert torch.norm(noisy_position - clean_position) > 0
        
    def test_generate_trajectory(self, simulator):
        """Test generating a complete trajectory."""
        # Generate trajectory
        trajectory_length = 10
        states, sensor_data = simulator.generate_trajectory(trajectory_length)
        
        # Check shapes
        assert len(states) == trajectory_length
        assert len(sensor_data) == trajectory_length
        
        assert states[0].shape == (12,)
        assert 'imu' in sensor_data[0]
        assert 'camera' in sensor_data[0]
        assert 'depth' in sensor_data[0]
        assert 'position' in sensor_data[0]
        
        # Check different trajectory with seed
        states1, _ = simulator.generate_trajectory(trajectory_length, seed=42)
        states2, _ = simulator.generate_trajectory(trajectory_length, seed=42)
        states3, _ = simulator.generate_trajectory(trajectory_length, seed=43)
        
        # Same seed should give same trajectory
        for i in range(trajectory_length):
            assert torch.allclose(states1[i], states2[i])
            
        # Different seed should give different trajectory
        assert not torch.allclose(states1[-1], states3[-1])


class TestDataGeneration:
    
    def test_generate_simulation_data(self, tmp_path):
        """Test the generate_simulation_data function."""
        # Create output directory
        output_dir = tmp_path / 'simulation_data'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create config
        config = SimulationConfig(
            num_trajectories=2,
            trajectory_length=5,
            dt=0.02,
            add_sensor_noise=True,
            camera_noise_std=0.05,
            imu_noise_std=0.02,
            position_noise_std=0.1,
            depth_noise_std=0.03,
            domain_randomization=False,
            seed=42
        )
        
        # Generate data
        generate_simulation_data(config, output_dir)
        
        # Check that data was generated
        # Train directory should contain 80% of trajectories
        train_dir = output_dir / 'train'
        assert os.path.exists(train_dir)
        
        # Val directory should contain 20% of trajectories
        val_dir = output_dir / 'val'
        assert os.path.exists(val_dir)
        
        # Check that config was saved
        assert os.path.exists(output_dir / 'config.json')
        
        # Check that files were created
        # For 2 trajectories with 5 steps each, split 80/20, we should have:
        # 8 samples in train (1 trajectory * 5 steps + 3 steps from second trajectory)
        # 2 samples in val (2 steps from second trajectory)
        assert len(os.listdir(train_dir)) > 0
        assert len(os.listdir(val_dir)) > 0
        
        # Check for specific files
        assert os.path.exists(train_dir / 'state_00000000.npy')
        assert os.path.exists(train_dir / 'imu_00000000.npy')
        assert os.path.exists(train_dir / 'camera_00000000.png')
        assert os.path.exists(train_dir / 'depth_00000000.npy')
        assert os.path.exists(train_dir / 'position_00000000.npy') 