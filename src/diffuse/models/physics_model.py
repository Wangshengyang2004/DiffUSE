import torch
import numpy as np

class UAVPhysicsModel:
    """
    Physics model for enforcing constraints on UAV state predictions.
    
    This model implements dynamics constraints, enforces physical limitations,
    and ensures the generated state estimates are physically plausible.
    """
    def __init__(
        self,
        dt=0.01,  # Time step in seconds
        mass=1.0,  # UAV mass in kg
        max_velocity=10.0,  # Maximum linear velocity in m/s
        max_acceleration=20.0,  # Maximum linear acceleration in m/s^2
        max_angular_velocity=5.0,  # Maximum angular velocity in rad/s
        max_angular_acceleration=20.0,  # Maximum angular acceleration in rad/s^2
        gravity=9.81,  # Gravity constant in m/s^2
        inertia_tensor=None,  # Inertia tensor (3x3) of the UAV
    ):
        """
        Initialize the UAV physics model.
        
        Args:
            dt: Time step in seconds
            mass: UAV mass in kg
            max_velocity: Maximum linear velocity in m/s
            max_acceleration: Maximum linear acceleration in m/s^2
            max_angular_velocity: Maximum angular velocity in rad/s
            max_angular_acceleration: Maximum angular acceleration in rad/s^2
            gravity: Gravity constant in m/s^2
            inertia_tensor: Inertia tensor (3x3) of the UAV
        """
        self.dt = dt
        self.mass = mass
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_angular_velocity = max_angular_velocity
        self.max_angular_acceleration = max_angular_acceleration
        self.gravity = gravity
        
        # Default inertia tensor for a symmetric UAV if not provided
        if inertia_tensor is None:
            self.inertia_tensor = torch.tensor([
                [0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.02]
            ], dtype=torch.float32)
        else:
            self.inertia_tensor = inertia_tensor
            
    def enforce_constraints(self, state):
        """
        Enforce physical constraints on UAV state.
        
        Args:
            state: UAV state tensor [batch_size, state_dim]
                state dim = 12:
                - position (3)
                - orientation (3, as Euler angles or quaternion depending on implementation)
                - linear velocity (3)
                - angular velocity (3)
                
        Returns:
            Physically constrained state
        """
        # Clone state to avoid modifying the original
        constrained_state = state.clone()
        
        # Split state into components
        position = constrained_state[:, 0:3]
        orientation = constrained_state[:, 3:6]  # Euler angles (roll, pitch, yaw)
        linear_velocity = constrained_state[:, 6:9]
        angular_velocity = constrained_state[:, 9:12]
        
        # 1. Enforce orientation constraints (normalize Euler angles)
        # Wrap angles to [-π, π]
        orientation = torch.atan2(torch.sin(orientation), torch.cos(orientation))
        constrained_state[:, 3:6] = orientation
        
        # 2. Enforce velocity constraints
        # Clip linear velocity magnitude
        linear_velocity_norm = torch.norm(linear_velocity, dim=1, keepdim=True)
        linear_velocity_normalized = linear_velocity / (linear_velocity_norm + 1e-8)
        linear_velocity_clipped = linear_velocity_normalized * torch.clamp(linear_velocity_norm, max=self.max_velocity)
        constrained_state[:, 6:9] = linear_velocity_clipped
        
        # Clip angular velocity magnitude
        angular_velocity_norm = torch.norm(angular_velocity, dim=1, keepdim=True)
        angular_velocity_normalized = angular_velocity / (angular_velocity_norm + 1e-8)
        angular_velocity_clipped = angular_velocity_normalized * torch.clamp(angular_velocity_norm, max=self.max_angular_velocity)
        constrained_state[:, 9:12] = angular_velocity_clipped
        
        return constrained_state
    
    def simulate_step(self, state, control_inputs=None):
        """
        Simulate one step of UAV dynamics.
        
        Args:
            state: Current UAV state [batch_size, state_dim]
            control_inputs: Control inputs [batch_size, control_dim]
                For a typical quadrotor, control_dim = 4:
                - Total thrust
                - Roll moment
                - Pitch moment
                - Yaw moment
                
        Returns:
            Next state after time step dt
        """
        # If no control inputs provided, assume zero controls
        if control_inputs is None:
            batch_size = state.shape[0]
            device = state.device
            control_inputs = torch.zeros(batch_size, 4, device=device)
            
        # Extract state components
        position = state[:, 0:3]
        orientation = state[:, 3:6]  # Euler angles (roll, pitch, yaw)
        linear_velocity = state[:, 6:9]
        angular_velocity = state[:, 9:12]
        
        # Extract control inputs
        thrust = control_inputs[:, 0:1]  # Total thrust magnitude
        moments = control_inputs[:, 1:4]  # Roll, pitch, yaw moments
        
        # 1. Calculate orientation as rotation matrix (from Euler angles)
        # This is a simplified version - in practice, use quaternions for better numerical stability
        sin_roll = torch.sin(orientation[:, 0:1])
        cos_roll = torch.cos(orientation[:, 0:1])
        sin_pitch = torch.sin(orientation[:, 1:2])
        cos_pitch = torch.cos(orientation[:, 1:2])
        sin_yaw = torch.sin(orientation[:, 2:3])
        cos_yaw = torch.cos(orientation[:, 2:3])
        
        # Basic rotation matrix (simplified)
        R_z = torch.cat([
            torch.cat([cos_yaw, -sin_yaw, torch.zeros_like(cos_yaw)], dim=1).unsqueeze(1),
            torch.cat([sin_yaw, cos_yaw, torch.zeros_like(cos_yaw)], dim=1).unsqueeze(1),
            torch.cat([torch.zeros_like(cos_yaw), torch.zeros_like(cos_yaw), torch.ones_like(cos_yaw)], dim=1).unsqueeze(1)
        ], dim=1)
        
        R_y = torch.cat([
            torch.cat([cos_pitch, torch.zeros_like(cos_pitch), sin_pitch], dim=1).unsqueeze(1),
            torch.cat([torch.zeros_like(cos_pitch), torch.ones_like(cos_pitch), torch.zeros_like(cos_pitch)], dim=1).unsqueeze(1),
            torch.cat([-sin_pitch, torch.zeros_like(cos_pitch), cos_pitch], dim=1).unsqueeze(1)
        ], dim=1)
        
        R_x = torch.cat([
            torch.cat([torch.ones_like(cos_roll), torch.zeros_like(cos_roll), torch.zeros_like(cos_roll)], dim=1).unsqueeze(1),
            torch.cat([torch.zeros_like(cos_roll), cos_roll, -sin_roll], dim=1).unsqueeze(1),
            torch.cat([torch.zeros_like(cos_roll), sin_roll, cos_roll], dim=1).unsqueeze(1)
        ], dim=1)
        
        # Full rotation matrix (from body to world frame)
        # R = R_z @ R_y @ R_x  # More accurate but more complex
        
        # 2. Calculate linear acceleration in world frame
        # Gravity force in world frame
        gravity_force = torch.zeros_like(position)
        gravity_force[:, 2] = -self.gravity
        
        # Thrust force in body frame (aligned with z-axis)
        thrust_vector = torch.zeros_like(position)
        thrust_vector[:, 2] = thrust.squeeze(-1) / self.mass
        
        # Convert thrust to world frame using rotation matrix
        # This is a simplification - in a real implementation, use proper matrix multiplication
        # thrust_world = R @ thrust_vector
        
        # Simplification: align thrust with negative gravity for hovering
        thrust_world = thrust_vector
        
        # Total acceleration = gravity + thrust
        acceleration = gravity_force + thrust_world
        
        # Clip acceleration to physical limits
        acceleration_norm = torch.norm(acceleration, dim=1, keepdim=True)
        acceleration_normalized = acceleration / (acceleration_norm + 1e-8)
        acceleration = acceleration_normalized * torch.clamp(acceleration_norm, max=self.max_acceleration)
        
        # 3. Calculate angular acceleration in body frame
        # angular_acceleration = torch.inverse(self.inertia_tensor) @ (moments - torch.cross(angular_velocity, self.inertia_tensor @ angular_velocity))
        # Simplified version:
        angular_acceleration = moments / 0.01  # Approximate using moment of inertia
        
        # Clip angular acceleration
        angular_acc_norm = torch.norm(angular_acceleration, dim=1, keepdim=True)
        angular_acc_normalized = angular_acceleration / (angular_acc_norm + 1e-8)
        angular_acceleration = angular_acc_normalized * torch.clamp(angular_acc_norm, max=self.max_angular_acceleration)
        
        # 4. Update state using Euler integration
        # Update position
        new_position = position + linear_velocity * self.dt
        
        # Update linear velocity
        new_linear_velocity = linear_velocity + acceleration * self.dt
        
        # Update orientation 
        new_orientation = orientation + angular_velocity * self.dt
        
        # Update angular velocity
        new_angular_velocity = angular_velocity + angular_acceleration * self.dt
        
        # 5. Assemble new state
        new_state = torch.cat([
            new_position,
            new_orientation,
            new_linear_velocity,
            new_angular_velocity
        ], dim=1)
        
        return new_state
    
    def consistency_score(self, state_t, state_t_plus_1):
        """
        Calculate physical consistency score between consecutive states.
        
        Args:
            state_t: State at time t
            state_t_plus_1: State at time t+1
            
        Returns:
            Consistency score (higher is better, negative indicates physically implausible)
        """
        # Simulate dynamics from state_t
        simulated_next_state = self.simulate_step(state_t)
        
        # Compare with provided state_t_plus_1
        position_error = torch.norm(simulated_next_state[:, 0:3] - state_t_plus_1[:, 0:3], dim=1)
        orientation_error = torch.norm(simulated_next_state[:, 3:6] - state_t_plus_1[:, 3:6], dim=1)
        velocity_error = torch.norm(simulated_next_state[:, 6:9] - state_t_plus_1[:, 6:9], dim=1)
        angular_vel_error = torch.norm(simulated_next_state[:, 9:12] - state_t_plus_1[:, 9:12], dim=1)
        
        # Weight the errors (adjust weights as needed)
        weighted_error = (
            1.0 * position_error + 
            0.5 * orientation_error + 
            0.8 * velocity_error + 
            0.3 * angular_vel_error
        )
        
        # Convert to a score (higher is better)
        # Exponential decay of score with error
        consistency_score = torch.exp(-weighted_error)
        
        return consistency_score 