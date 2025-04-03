"""
Tests for diffusion process components.
"""
import pytest
import torch
import numpy as np

from src.diffuse.models.diffusion_process import DiffusionProcess


class TestDiffusionProcess:
    
    def test_init(self, diffusion_model):
        """Test DiffusionProcess initialization."""
        diffusion = DiffusionProcess(
            model=diffusion_model,
            n_timesteps=1000,
            beta_schedule='linear',
            beta_start=1e-4,
            beta_end=0.02
        )
        
        assert diffusion.model == diffusion_model
        assert diffusion.n_timesteps == 1000
        assert isinstance(diffusion.betas, torch.Tensor)
        assert diffusion.betas.shape == (1000,)
        
        # Check precomputed values
        assert isinstance(diffusion.alphas, torch.Tensor)
        assert isinstance(diffusion.alphas_cumprod, torch.Tensor)
        assert isinstance(diffusion.alphas_cumprod_prev, torch.Tensor)
        assert isinstance(diffusion.sqrt_alphas_cumprod, torch.Tensor)
        assert isinstance(diffusion.sqrt_one_minus_alphas_cumprod, torch.Tensor)
        assert isinstance(diffusion.posterior_variance, torch.Tensor)
        
    def test_init_with_different_schedules(self, diffusion_model):
        """Test DiffusionProcess initialization with different beta schedules."""
        # Test cosine schedule
        diffusion_cosine = DiffusionProcess(
            model=diffusion_model,
            n_timesteps=100,
            beta_schedule='cosine',
            beta_start=1e-4,
            beta_end=0.02
        )
        
        assert diffusion_cosine.betas.shape == (100,)
        assert torch.all(diffusion_cosine.betas >= 0.0001)
        assert torch.all(diffusion_cosine.betas <= 0.9999)
        
        # Test quadratic schedule
        diffusion_quadratic = DiffusionProcess(
            model=diffusion_model,
            n_timesteps=100,
            beta_schedule='quadratic',
            beta_start=1e-4,
            beta_end=0.02
        )
        
        assert diffusion_quadratic.betas.shape == (100,)
        assert torch.isclose(diffusion_quadratic.betas[0], torch.tensor(1e-4))
        assert torch.isclose(diffusion_quadratic.betas[-1], torch.tensor(0.02))
        
    def test_add_noise(self, diffusion_process, device):
        """Test adding noise to a clean state."""
        batch_size = 2
        state_dim = 12
        
        # Create a clean state
        x_0 = torch.rand(batch_size, state_dim, device=device)
        
        # Select timesteps
        t = torch.tensor([0, 5], dtype=torch.long, device=device)
        
        # Add noise
        x_t, noise = diffusion_process.add_noise(x_0, t)
        
        # Check shapes
        assert x_t.shape == (batch_size, state_dim)
        assert noise.shape == (batch_size, state_dim)
        
        # Check values
        assert not torch.isnan(x_t).any()
        assert not torch.isnan(noise).any()
        
        # At t=0, x_t should be very close to x_0
        t_zero = torch.tensor([0, 0], dtype=torch.long, device=device)
        x_t_zero, _ = diffusion_process.add_noise(x_0, t_zero)
        assert torch.allclose(x_t_zero, x_0, atol=1e-5)
        
        # At t=n_timesteps-1, x_t should be dominated by noise
        t_max = torch.tensor([diffusion_process.n_timesteps-1, diffusion_process.n_timesteps-1], 
                             dtype=torch.long, device=device)
        x_t_max, noise_max = diffusion_process.add_noise(x_0, t_max)
        assert torch.norm(x_t_max - noise_max) < torch.norm(x_t_max - x_0)
        
    def test_reverse_step(self, diffusion_process, device, sample_sensor_data):
        """Test single reverse diffusion step."""
        batch_size = 2
        state_dim = 12
        
        # Create a noisy state
        x_t = torch.rand(batch_size, state_dim, device=device)
        
        # Select timestep
        t = torch.tensor([5, 5], dtype=torch.long, device=device)
        
        # Move sensor data to device
        sensor_data = {k: v.to(device) for k, v in sample_sensor_data.items()}
        
        # Perform reverse step
        x_t_minus_1 = diffusion_process.reverse_step(x_t, t, sensor_data)
        
        # Check shape
        assert x_t_minus_1.shape == (batch_size, state_dim)
        
        # Check values
        assert not torch.isnan(x_t_minus_1).any()
        
    def test_sample(self, diffusion_process, device, sample_sensor_data):
        """Test sampling from the diffusion process."""
        batch_size = 2
        state_dim = 12
        
        # Create initial state
        initial_state = torch.zeros(batch_size, state_dim, device=device)
        
        # Move sensor data to device
        sensor_data = {k: v.to(device) for k, v in sample_sensor_data.items()}
        
        # Sample from diffusion process
        estimated_state = diffusion_process.sample(
            initial_state=initial_state,
            sensor_data=sensor_data,
            device=device,
            show_progress=False
        )
        
        # Check shape
        assert estimated_state.shape == (batch_size, state_dim)
        
        # Check values
        assert not torch.isnan(estimated_state).any()
        
    def test_physical_corrected_sample(self, diffusion_process, physics_model, device, sample_sensor_data):
        """Test sampling with physical constraints."""
        batch_size = 2
        state_dim = 12
        
        # Create initial state
        initial_state = torch.zeros(batch_size, state_dim, device=device)
        
        # Move sensor data to device
        sensor_data = {k: v.to(device) for k, v in sample_sensor_data.items()}
        
        # Sample with physical constraints
        estimated_state = diffusion_process.physical_corrected_sample(
            initial_state=initial_state,
            sensor_data=sensor_data,
            physics_model=physics_model,
            device=device,
            show_progress=False
        )
        
        # Check shape
        assert estimated_state.shape == (batch_size, state_dim)
        
        # Check values
        assert not torch.isnan(estimated_state).any()
        
    def test_train_step(self, diffusion_process, device, sample_uav_state, sample_sensor_data):
        """Test training step."""
        # Create clean state
        clean_state = sample_uav_state
        
        # Create optimizer
        optimizer = torch.optim.Adam(diffusion_process.model.parameters(), lr=1e-4)
        
        # Execute training step
        loss = diffusion_process.train_step(
            clean_state=clean_state,
            sensor_data=sample_sensor_data,
            optimizer=optimizer,
            device=device
        )
        
        # Check loss is a scalar
        assert isinstance(loss, float)
        
        # Check loss is not NaN
        assert not np.isnan(loss) 