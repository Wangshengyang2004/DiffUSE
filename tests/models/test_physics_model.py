"""
Tests for physics model components.
"""
import pytest
import torch
import torch.nn as nn

from src.diffuse.models.physics_model import UAVPhysicsModel


class TestUAVPhysicsModel:
    
    def test_init(self):
        """Test UAVPhysicsModel initialization."""
        model = UAVPhysicsModel(
            dt=0.01,
            mass=1.5,
            max_velocity=8.0,
            max_acceleration=15.0,
            max_angular_velocity=4.0,
            max_angular_acceleration=18.0,
            gravity=9.81
        )
        
        assert model.dt == 0.01
        assert model.mass == 1.5
        assert model.max_velocity == 8.0
        assert model.max_acceleration == 15.0
        assert model.max_angular_velocity == 4.0
        assert model.max_angular_acceleration == 18.0
        assert model.gravity == 9.81
        assert isinstance(model.inertia_tensor, torch.Tensor)
        assert model.inertia_tensor.shape == (3, 3)
        
    def test_init_with_custom_inertia(self):
        """Test UAVPhysicsModel initialization with custom inertia tensor."""
        custom_inertia = torch.tensor([
            [0.02, 0, 0],
            [0, 0.02, 0],
            [0, 0, 0.03]
        ], dtype=torch.float32)
        
        model = UAVPhysicsModel(inertia_tensor=custom_inertia)
        
        assert torch.allclose(model.inertia_tensor, custom_inertia)
        
    def test_enforce_constraints(self):
        """Test enforcing physical constraints on UAV state."""
        batch_size = 2
        state_dim = 12
        
        # Create physics model
        model = UAVPhysicsModel(
            max_velocity=10.0,
            max_angular_velocity=5.0
        )
        
        # Create state with excessive velocities
        state = torch.zeros(batch_size, state_dim)
        
        # Set excessive linear velocities
        state[:, 6:9] = torch.tensor([[-15.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        
        # Set excessive angular velocities
        state[:, 9:12] = torch.tensor([[0.0, -10.0, 0.0], [8.0, 0.0, 0.0]])
        
        # Enforce constraints
        constrained_state = model.enforce_constraints(state)
        
        # Check linear velocities are clipped
        linear_velocity_norm = torch.norm(constrained_state[:, 6:9], dim=1)
        assert torch.all(linear_velocity_norm <= model.max_velocity + 1e-6)
        
        # Check angular velocities are clipped
        angular_velocity_norm = torch.norm(constrained_state[:, 9:12], dim=1)
        assert torch.all(angular_velocity_norm <= model.max_angular_velocity + 1e-6)
        
        # Check orientation angles are normalized
        orientation = constrained_state[:, 3:6]
        assert torch.all(orientation >= -torch.pi - 1e-6)
        assert torch.all(orientation <= torch.pi + 1e-6)
        
    def test_simulate_step(self):
        """Test simulating one step of UAV dynamics."""
        batch_size = 2
        state_dim = 12
        
        # Create physics model
        model = UAVPhysicsModel(dt=0.1)
        
        # Create initial state
        state = torch.zeros(batch_size, state_dim)
        
        # Set position
        state[:, 0:3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        # Set orientation
        state[:, 3:6] = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        # Set linear velocity
        state[:, 6:9] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        # Set angular velocity
        state[:, 9:12] = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        
        # Simulate step with zero control inputs
        next_state = model.simulate_step(state)
        
        # Check shape
        assert next_state.shape == (batch_size, state_dim)
        
        # Check position increased in the direction of velocity
        assert torch.all(next_state[:, 0] > state[:, 0])
        assert torch.all(next_state[:, 1] > state[:, 1])
        
        # Check orientation changed due to angular velocity
        assert not torch.allclose(next_state[:, 3:6], state[:, 3:6])
        
        # Check linear velocity changed due to gravity
        assert not torch.allclose(next_state[:, 6:9], state[:, 6:9])
        
        # Check angular velocity stays the same (no moments)
        assert torch.allclose(next_state[:, 9:12], state[:, 9:12])
        
    def test_simulate_step_with_control(self):
        """Test simulating one step with control inputs."""
        batch_size = 2
        state_dim = 12
        
        # Create physics model
        model = UAVPhysicsModel(dt=0.1)
        
        # Create initial state
        state = torch.zeros(batch_size, state_dim)
        
        # Create control inputs
        # Format: [thrust, roll_moment, pitch_moment, yaw_moment]
        control = torch.zeros(batch_size, 4)
        
        # Set thrust to counteract gravity
        control[:, 0] = model.mass * model.gravity
        
        # Set some roll moment
        control[:, 1] = 0.1
        
        # Simulate step with control inputs
        next_state = model.simulate_step(state, control)
        
        # Check shape
        assert next_state.shape == (batch_size, state_dim)
        
        # Check angular velocity changed due to moments
        assert not torch.allclose(next_state[:, 9], state[:, 9])
        
    def test_consistency_score(self):
        """Test physical consistency score between consecutive states."""
        batch_size = 2
        state_dim = 12
        
        # Create physics model
        model = UAVPhysicsModel(dt=0.1)
        
        # Create initial state
        state_t = torch.zeros(batch_size, state_dim)
        
        # Set some non-zero values
        state_t[:, 0:3] = 1.0  # position
        state_t[:, 6:9] = 0.5  # linear velocity
        
        # Generate next state using physics model
        state_t_plus_1 = model.simulate_step(state_t)
        
        # Calculate consistency score
        score = model.consistency_score(state_t, state_t_plus_1)
        
        # Check score shape
        assert score.shape == (batch_size,)
        
        # Score should be high (close to 1) for physically consistent states
        assert torch.all(score > 0.9)
        
        # Create physically inconsistent state by modifying the simulated next state
        inconsistent_state = state_t_plus_1.clone()
        inconsistent_state[:, 0:3] += 10.0  # Large position change
        
        # Calculate consistency score for inconsistent state
        inconsistent_score = model.consistency_score(state_t, inconsistent_state)
        
        # Score should be low for physically inconsistent states
        assert torch.all(inconsistent_score < score) 