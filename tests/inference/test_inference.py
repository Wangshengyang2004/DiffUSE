"""
Tests for inference module.
"""
import pytest
import torch
import numpy as np
import os
from pathlib import Path

from src.diffuse.models.physics_model import UAVPhysicsModel
from src.diffuse.models.diffusion_model import DiffusionModel, DenoisingUNet
from src.diffuse.inference.inference import DiffuseInference, DiffuseConfig


class TestDiffuseConfig:
    
    def test_init_from_dict(self):
        """Test initializing config from dictionary."""
        config_dict = {
            'model_path': 'path/to/model.pt',
            'state_dim': 12,
            'condition_dim': 64,
            'beta_schedule': 'linear',
            'timesteps': 1000,
            'sample_steps': 50,
            'guidance_scale': 1.0,
            'use_physics_model': True,
            'consistency_weight': 0.5,
            'device': 'cpu'
        }
        
        config = DiffuseConfig.from_dict(config_dict)
        
        assert config.model_path == 'path/to/model.pt'
        assert config.state_dim == 12
        assert config.condition_dim == 64
        assert config.beta_schedule == 'linear'
        assert config.timesteps == 1000
        assert config.sample_steps == 50
        assert config.guidance_scale == 1.0
        assert config.use_physics_model
        assert config.consistency_weight == 0.5
        assert config.device == 'cpu'
        
    def test_init_from_file(self, tmp_path):
        """Test initializing config from a file."""
        config_dict = {
            'model_path': 'path/to/model.pt',
            'state_dim': 12,
            'condition_dim': 64,
            'beta_schedule': 'linear',
            'timesteps': 1000,
            'sample_steps': 50,
            'guidance_scale': 1.0,
            'use_physics_model': True,
            'consistency_weight': 0.5,
            'device': 'cpu'
        }
        
        # Save config to file
        import json
        config_path = tmp_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f)
            
        # Load config from file
        config = DiffuseConfig.from_file(config_path)
        
        assert config.model_path == 'path/to/model.pt'
        assert config.state_dim == 12
        assert config.condition_dim == 64
        
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = DiffuseConfig(
            model_path='path/to/model.pt',
            state_dim=12,
            condition_dim=64,
            beta_schedule='linear',
            timesteps=1000,
            sample_steps=50,
            guidance_scale=1.0,
            use_physics_model=True,
            consistency_weight=0.5,
            device='cpu'
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['model_path'] == 'path/to/model.pt'
        assert config_dict['state_dim'] == 12
        assert config_dict['condition_dim'] == 64
        assert config_dict['beta_schedule'] == 'linear'
        assert config_dict['timesteps'] == 1000
        assert config_dict['sample_steps'] == 50
        assert config_dict['guidance_scale'] == 1.0
        assert config_dict['use_physics_model'] is True
        assert config_dict['consistency_weight'] == 0.5
        assert config_dict['device'] == 'cpu'


class TestDiffuseInference:
    
    @pytest.fixture
    def inference_setup(self, tmp_path):
        """Set up inference components for testing."""
        # Create a simple model and save it
        model = DiffusionModel(
            state_dim=12,
            condition_dim=64,
            denoise_net=DenoisingUNet(
                state_dim=12,
                condition_dim=64,
                hidden_dims=[128, 256, 128],
                time_embed_dim=32
            ),
            timesteps=100,
            beta_schedule='linear'
        )
        
        # Save model checkpoint
        model_path = tmp_path / 'model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
        }, model_path)
        
        # Create physics model
        physics_model = UAVPhysicsModel()
        
        # Create config
        config = DiffuseConfig(
            model_path=str(model_path),
            state_dim=12,
            condition_dim=64,
            beta_schedule='linear',
            timesteps=100,
            sample_steps=10,
            guidance_scale=1.0,
            use_physics_model=True,
            consistency_weight=0.5,
            device='cpu'
        )
        
        # Create inference object
        inference = DiffuseInference(config)
        
        return {
            'inference': inference,
            'config': config,
            'model': model,
            'physics_model': physics_model,
            'model_path': model_path
        }
    
    def test_init(self, inference_setup):
        """Test initialization of inference module."""
        inference = inference_setup['inference']
        config = inference_setup['config']
        
        assert inference.config is config
        assert isinstance(inference.diffusion_model, DiffusionModel)
        assert isinstance(inference.physics_model, UAVPhysicsModel)
        
    def test_load_model(self, inference_setup):
        """Test loading model from checkpoint."""
        inference = inference_setup['inference']
        model_path = inference_setup['model_path']
        
        # Load model again
        inference.load_model(model_path)
        
        # Check model structure
        assert isinstance(inference.diffusion_model, DiffusionModel)
        assert inference.diffusion_model.state_dim == 12
        assert inference.diffusion_model.condition_dim == 64
        
    def test_predict_single(self, inference_setup):
        """Test predicting a single state."""
        inference = inference_setup['inference']
        
        # Create mock sensor data
        sensor_data = {
            'combined': torch.randn(64)
        }
        
        # Mock the model's sample method to return a fixed state
        fixed_state = torch.ones(12)
        
        def mock_sample(condition, sample_steps, guidance_scale=1.0):
            assert condition.shape == (1, 64)
            assert sample_steps == 10
            assert guidance_scale == 1.0
            return fixed_state.unsqueeze(0)
            
        inference.diffusion_model.sample = mock_sample
        
        # Predict state
        predicted_state = inference.predict(sensor_data)
        
        # Check output
        assert torch.allclose(predicted_state, fixed_state)
        
    def test_predict_batch(self, inference_setup):
        """Test predicting a batch of states."""
        inference = inference_setup['inference']
        
        # Create mock batch of sensor data
        batch_size = 4
        sensor_data = {
            'combined': torch.randn(batch_size, 64)
        }
        
        # Mock the model's sample method to return fixed states
        fixed_states = torch.ones(batch_size, 12)
        
        def mock_sample(condition, sample_steps, guidance_scale=1.0):
            assert condition.shape == (batch_size, 64)
            assert sample_steps == 10
            assert guidance_scale == 1.0
            return fixed_states
            
        inference.diffusion_model.sample = mock_sample
        
        # Predict states
        predicted_states = inference.predict(sensor_data)
        
        # Check output
        assert torch.allclose(predicted_states, fixed_states)
        
    def test_predict_with_physics_refinement(self, inference_setup):
        """Test prediction with physics-based refinement."""
        inference = inference_setup['inference']
        
        # Set consistency weight
        inference.config.consistency_weight = 0.5
        
        # Create mock sequence of sensor data
        seq_length = 3
        batch_size = 2
        
        # 3D sequence: [sequence, batch, features]
        sensor_data = {
            'combined': torch.randn(seq_length, batch_size, 64)
        }
        
        # Mock the model's sample method to return states that need refinement
        states_to_refine = torch.randn(seq_length, batch_size, 12)
        
        def mock_sample(condition, sample_steps, guidance_scale=1.0):
            # Return a different state for each sequence step
            if condition.dim() == 2:
                # Single step prediction
                seq_idx = inference._current_seq_idx
                return states_to_refine[seq_idx]
            else:
                # Full sequence at once
                return states_to_refine.reshape(-1, 12)
                
        inference.diffusion_model.sample = mock_sample
        
        # Add attribute to track the current sequence index
        inference._current_seq_idx = 0
        
        # Mock the physics model to just return the states
        def mock_consistency_score(states):
            # Return a dummy score
            return torch.tensor(0.8)
            
        inference.physics_model.compute_consistency_score = mock_consistency_score
        
        # Predict with physics refinement
        inference.config.use_physics_model = True
        
        # Call the predict method for sequence
        predicted_states = inference.predict_sequence(sensor_data)
        
        # Check output shape
        assert predicted_states.shape == (seq_length, batch_size, 12)
        
    def test_predict_sequence(self, inference_setup):
        """Test predicting a sequence of states."""
        inference = inference_setup['inference']
        
        # Create mock sequence of sensor data
        seq_length = 3
        batch_size = 2
        
        # 3D sequence: [sequence, batch, features]
        sensor_data = {
            'combined': torch.randn(seq_length, batch_size, 64)
        }
        
        # Mock the model's sample method to return fixed states
        fixed_states = torch.ones(seq_length * batch_size, 12)
        
        def mock_sample(condition, sample_steps, guidance_scale=1.0):
            if condition.dim() == 2:
                # Single step prediction (for sequential processing)
                return fixed_states[:batch_size]
            else:
                # Batched sequence (all steps at once)
                return fixed_states
                
        inference.diffusion_model.sample = mock_sample
        
        # Predict states for the sequence
        predicted_states = inference.predict_sequence(sensor_data)
        
        # Check output shape
        assert predicted_states.shape == (seq_length, batch_size, 12)
        assert torch.all(predicted_states == 1.0) 