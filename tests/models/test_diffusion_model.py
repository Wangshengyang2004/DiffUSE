"""
Tests for diffusion model components.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.diffuse.models.diffusion_model import (
    TimeEmbedding,
    SensorEmbedding,
    UNetBlock,
    DiffusionStateEstimator,
    DiffusionModel,
    DenoisingUNet
)


class TestTimeEmbedding:
    
    def test_init(self):
        """Test TimeEmbedding initialization."""
        emb = TimeEmbedding(dim=32)
        assert isinstance(emb, nn.Module)
        assert emb.dim == 32
        assert isinstance(emb.proj, nn.Sequential)
        
    def test_forward(self):
        """Test TimeEmbedding forward pass."""
        batch_size = 2
        dim = 32
        
        emb = TimeEmbedding(dim=dim)
        t = torch.tensor([0, 10], dtype=torch.long)
        
        output = emb(t)
        
        # Check output shape
        assert output.shape == (batch_size, dim * 4)
        # Check output is not NaN
        assert not torch.isnan(output).any()


class TestSensorEmbedding:
    
    def test_init(self):
        """Test SensorEmbedding initialization."""
        sensor_channels = {'imu': 6, 'camera': 3, 'depth': 1, 'position': 3}
        embedding_dim = 64
        
        emb = SensorEmbedding(sensor_channels, embedding_dim)
        
        assert isinstance(emb, nn.Module)
        assert emb.sensor_channels == sensor_channels
        assert emb.embedding_dim == embedding_dim
        assert isinstance(emb.imu_encoder, nn.Sequential)
        assert isinstance(emb.camera_encoder, nn.Sequential)
        assert isinstance(emb.depth_encoder, nn.Sequential)
        assert isinstance(emb.position_encoder, nn.Sequential)
        assert isinstance(emb.fusion, nn.Sequential)
        
    def test_forward(self, sample_sensor_data):
        """Test SensorEmbedding forward pass."""
        sensor_channels = {'imu': 6, 'camera': 3, 'depth': 1, 'position': 3}
        embedding_dim = 64
        
        emb = SensorEmbedding(sensor_channels, embedding_dim)
        
        output = emb(sample_sensor_data)
        
        # Check output shape
        assert output.shape == (2, embedding_dim)
        # Check output is not NaN
        assert not torch.isnan(output).any()
        
    def test_forward_with_missing_sensors(self):
        """Test SensorEmbedding forward pass with missing sensors."""
        sensor_channels = {'imu': 6, 'camera': 3, 'depth': 1, 'position': 3}
        embedding_dim = 64
        
        emb = SensorEmbedding(sensor_channels, embedding_dim)
        
        # Create sample data with only IMU and position
        partial_data = {
            'imu': torch.rand(2, 6),
            'position': torch.rand(2, 3)
        }
        
        output = emb(partial_data)
        
        # Check output shape
        assert output.shape == (2, embedding_dim)
        # Check output is not NaN
        assert not torch.isnan(output).any()


class TestUNetBlock:
    
    def test_init(self):
        """Test UNetBlock initialization."""
        in_channels = 32
        out_channels = 64
        time_dim = 128
        
        block = UNetBlock(in_channels, out_channels, time_dim, use_attention=True)
        
        assert isinstance(block, nn.Module)
        assert isinstance(block.time_mlp, nn.Sequential)
        assert isinstance(block.conv1, nn.Conv2d)
        assert isinstance(block.conv2, nn.Conv2d)
        assert isinstance(block.attention, nn.MultiheadAttention)
        
    def test_forward(self):
        """Test UNetBlock forward pass."""
        batch_size = 2
        in_channels = 32
        out_channels = 64
        time_dim = 128
        height, width = 8, 8
        
        block = UNetBlock(in_channels, out_channels, time_dim, use_attention=False)
        
        # Create dummy inputs
        x = torch.rand(batch_size, in_channels, height, width)
        t = torch.rand(batch_size, time_dim)
        
        output = block(x, t)
        
        # Check output shape
        assert output.shape == (batch_size, out_channels, height, width)
        # Check output is not NaN
        assert not torch.isnan(output).any()
        
    def test_forward_with_attention(self):
        """Test UNetBlock forward pass with attention."""
        batch_size = 2
        in_channels = 32
        out_channels = 64
        time_dim = 128
        height, width = 8, 8
        
        block = UNetBlock(in_channels, out_channels, time_dim, use_attention=True)
        
        # Create dummy inputs
        x = torch.rand(batch_size, in_channels, height, width)
        t = torch.rand(batch_size, time_dim)
        
        output = block(x, t)
        
        # Check output shape
        assert output.shape == (batch_size, out_channels, height, width)
        # Check output is not NaN
        assert not torch.isnan(output).any()


class TestDiffusionStateEstimator:
    
    def test_init(self):
        """Test DiffusionStateEstimator initialization."""
        model = DiffusionStateEstimator(
            state_dim=12,
            time_embedding_dim=64,
            sensor_channels={'imu': 6, 'camera': 3, 'depth': 1, 'position': 3},
            sensor_embedding_dim=64,
            hidden_dims=[64, 128, 256, 128, 64],
            use_attention=True
        )
        
        assert isinstance(model, nn.Module)
        assert model.state_dim == 12
        assert isinstance(model.time_embedding, TimeEmbedding)
        assert isinstance(model.sensor_embedding, SensorEmbedding)
        assert isinstance(model.state_encoder, nn.Sequential)
        assert isinstance(model.blocks, nn.ModuleList)
        assert len(model.blocks) == 4  # 5 dims in hidden_dims, but first is just used for initial projection
        assert isinstance(model.final, nn.Sequential)
        
    def test_forward(self, device, sample_uav_state, sample_sensor_data):
        """Test DiffusionStateEstimator forward pass."""
        batch_size = 2
        state_dim = 12
        
        model = DiffusionStateEstimator(
            state_dim=state_dim,
            time_embedding_dim=32,
            sensor_channels={'imu': 6, 'camera': 3, 'depth': 1, 'position': 3},
            sensor_embedding_dim=32,
            hidden_dims=[32, 64, 32],
            use_attention=False
        ).to(device)
        
        noisy_state = sample_uav_state.to(device)
        timestep = torch.tensor([0, 5], dtype=torch.long).to(device)
        
        # Move all sensor data to device
        sensor_data = {k: v.to(device) for k, v in sample_sensor_data.items()}
        
        output = model(noisy_state, timestep, sensor_data)
        
        # Check output shape
        assert output.shape == (batch_size, state_dim)
        # Check output is not NaN
        assert not torch.isnan(output).any() 


class TestDenoisingUNet:
    
    def test_init(self):
        """Test initialization of DenoisingUNet."""
        model = DenoisingUNet(
            state_dim=12,
            condition_dim=64,
            hidden_dims=[128, 256, 128],
            time_embed_dim=32
        )
        
        assert isinstance(model, DenoisingUNet)
        assert isinstance(model, nn.Module)
        assert model.state_dim == 12
        assert model.condition_dim == 64
        assert model.hidden_dims == [128, 256, 128]
        assert model.time_embed_dim == 32
        
    def test_forward(self):
        """Test forward pass of DenoisingUNet."""
        batch_size = 8
        state_dim = 12
        condition_dim = 64
        
        # Create model
        model = DenoisingUNet(
            state_dim=state_dim,
            condition_dim=condition_dim,
            hidden_dims=[128, 256, 128],
            time_embed_dim=32
        )
        
        # Create inputs
        x = torch.randn(batch_size, state_dim)
        t = torch.randint(0, 1000, (batch_size,))
        condition = torch.randn(batch_size, condition_dim)
        
        # Forward pass
        output = model(x, t, condition)
        
        # Check output shape
        assert output.shape == (batch_size, state_dim)
        
    def test_time_embedding(self):
        """Test time embedding layer of DenoisingUNet."""
        model = DenoisingUNet(
            state_dim=12,
            condition_dim=64,
            hidden_dims=[128, 256, 128],
            time_embed_dim=32
        )
        
        # Create time tensor
        batch_size = 8
        t = torch.randint(0, 1000, (batch_size,))
        
        # Get time embedding
        time_embed = model.time_mlp(t)
        
        # Check output shape
        assert time_embed.shape == (batch_size, 32)
        
        # Test that different timesteps produce different embeddings
        t1 = torch.tensor([100])
        t2 = torch.tensor([500])
        
        embed1 = model.time_mlp(t1)
        embed2 = model.time_mlp(t2)
        
        assert not torch.allclose(embed1, embed2)


class TestDiffusionModel:
    
    def test_init(self):
        """Test initialization of DiffusionModel."""
        model = DiffusionModel(
            state_dim=12,
            condition_dim=64,
            denoise_net=DenoisingUNet(
                state_dim=12,
                condition_dim=64,
                hidden_dims=[128, 256, 128],
                time_embed_dim=32
            ),
            timesteps=1000,
            beta_schedule='linear'
        )
        
        assert isinstance(model, DiffusionModel)
        assert isinstance(model, nn.Module)
        assert model.state_dim == 12
        assert model.condition_dim == 64
        assert model.timesteps == 1000
        assert model.beta_schedule == 'linear'
        assert model.betas.shape == (1000,)
        assert model.alphas.shape == (1000,)
        assert model.alphas_cumprod.shape == (1000,)
        
    def test_noise_schedule(self):
        """Test noise schedule initialization for different schedules."""
        # Test linear schedule
        model_linear = DiffusionModel(
            state_dim=12,
            condition_dim=64,
            denoise_net=DenoisingUNet(
                state_dim=12,
                condition_dim=64,
                hidden_dims=[128, 256, 128],
                time_embed_dim=32
            ),
            timesteps=1000,
            beta_schedule='linear'
        )
        
        assert model_linear.betas[0] < model_linear.betas[-1]
        assert torch.all(model_linear.betas >= 0)
        assert torch.all(model_linear.betas <= 1)
        
        # Test cosine schedule
        model_cosine = DiffusionModel(
            state_dim=12,
            condition_dim=64,
            denoise_net=DenoisingUNet(
                state_dim=12,
                condition_dim=64,
                hidden_dims=[128, 256, 128],
                time_embed_dim=32
            ),
            timesteps=1000,
            beta_schedule='cosine'
        )
        
        assert torch.all(model_cosine.betas >= 0)
        assert torch.all(model_cosine.betas <= 1)
        
        # Test quadratic schedule
        model_quad = DiffusionModel(
            state_dim=12,
            condition_dim=64,
            denoise_net=DenoisingUNet(
                state_dim=12,
                condition_dim=64,
                hidden_dims=[128, 256, 128],
                time_embed_dim=32
            ),
            timesteps=1000,
            beta_schedule='quadratic'
        )
        
        assert torch.all(model_quad.betas >= 0)
        assert torch.all(model_quad.betas <= 1)
        
    def test_forward_diffusion(self):
        """Test forward diffusion process."""
        model = DiffusionModel(
            state_dim=12,
            condition_dim=64,
            denoise_net=DenoisingUNet(
                state_dim=12,
                condition_dim=64,
                hidden_dims=[128, 256, 128],
                time_embed_dim=32
            ),
            timesteps=1000,
            beta_schedule='linear'
        )
        
        # Create clean states
        batch_size = 8
        x_0 = torch.randn(batch_size, 12)
        
        # Apply forward diffusion at different timesteps
        t_early = torch.full((batch_size,), 100, dtype=torch.long)
        t_mid = torch.full((batch_size,), 500, dtype=torch.long)
        t_late = torch.full((batch_size,), 900, dtype=torch.long)
        
        # Get noisy samples at different timesteps
        x_noisy_early, noise_early = model.forward_diffusion(x_0, t_early)
        x_noisy_mid, noise_mid = model.forward_diffusion(x_0, t_mid)
        x_noisy_late, noise_late = model.forward_diffusion(x_0, t_late)
        
        # Check output shapes
        assert x_noisy_early.shape == x_0.shape
        assert noise_early.shape == x_0.shape
        
        # Check that noise level increases with timestep
        # Calculate mean absolute difference between noisy and clean
        diff_early = torch.abs(x_noisy_early - x_0).mean()
        diff_mid = torch.abs(x_noisy_mid - x_0).mean()
        diff_late = torch.abs(x_noisy_late - x_0).mean()
        
        assert diff_early < diff_mid < diff_late
        
    def test_reverse_diffusion_step(self):
        """Test a single step of reverse diffusion."""
        model = DiffusionModel(
            state_dim=12,
            condition_dim=64,
            denoise_net=DenoisingUNet(
                state_dim=12,
                condition_dim=64,
                hidden_dims=[128, 256, 128],
                time_embed_dim=32
            ),
            timesteps=1000,
            beta_schedule='linear'
        )
        
        # Create noisy state
        batch_size = 4
        x_t = torch.randn(batch_size, 12)
        condition = torch.randn(batch_size, 64)
        timestep = torch.tensor([500] * batch_size)
        
        # Mock the denoise_net prediction by returning x_t itself
        # This is equivalent to predicting zero noise
        model.denoise_net = lambda x, t, c: x
        
        # Perform reverse diffusion step
        x_t_minus_1 = model.reverse_diffusion_step(x_t, timestep, condition)
        
        # Check output shape
        assert x_t_minus_1.shape == x_t.shape
        
        # Since we're predicting zero noise, x_t_minus_1 should be less noisy than x_t
        # when t is large enough
        assert torch.norm(x_t_minus_1) < torch.norm(x_t)
        
    def test_sample(self):
        """Test sampling from the diffusion model."""
        # Create a simple mock denoise_net that always predicts zero
        class MockDenoise(nn.Module):
            def forward(self, x, t, condition):
                return torch.zeros_like(x)
        
        model = DiffusionModel(
            state_dim=12,
            condition_dim=64,
            denoise_net=MockDenoise(),
            timesteps=10,  # Use fewer timesteps for faster testing
            beta_schedule='linear'
        )
        
        # Create condition
        batch_size = 4
        condition = torch.randn(batch_size, 64)
        
        # Sample from the model
        samples = model.sample(condition, sample_steps=10)
        
        # Check output shape
        assert samples.shape == (batch_size, 12)
        
        # With our mock network, the samples should converge toward zero
        assert torch.norm(samples) < torch.norm(torch.randn(batch_size, 12))
        
    def test_loss_calculation(self):
        """Test loss calculation for training."""
        model = DiffusionModel(
            state_dim=12,
            condition_dim=64,
            denoise_net=DenoisingUNet(
                state_dim=12,
                condition_dim=64,
                hidden_dims=[128, 256, 128],
                time_embed_dim=32
            ),
            timesteps=1000,
            beta_schedule='linear'
        )
        
        # Create inputs
        batch_size = 8
        x_0 = torch.randn(batch_size, 12)
        condition = torch.randn(batch_size, 64)
        
        # Mock the denoise_net to return fixed values
        model.denoise_net = lambda x, t, c: torch.zeros_like(x)
        
        # Calculate loss
        loss = model.compute_loss(x_0, condition)
        
        # Check that loss is a scalar
        assert loss.dim() == 0
        assert loss.requires_grad 