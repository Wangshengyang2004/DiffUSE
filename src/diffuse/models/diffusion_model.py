import torch
import torch.nn as nn
import numpy as np

class TimeEmbedding(nn.Module):
    """Time embedding for diffusion models."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)  # Changed to output dim instead of dim * 4
        )
        
    def forward(self, t):
        # Create sinusoidal position embeddings (based on transformer timing signals)
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        
        # Transform embeddings through MLP
        embeddings = self.proj(embeddings)
        return embeddings

class SensorEmbedding(nn.Module):
    """Encoder for multi-modal sensor inputs."""
    def __init__(self, sensor_channels, embedding_dim):
        super().__init__()
        self.sensor_channels = sensor_channels
        self.embedding_dim = embedding_dim
        
        # Sensor-specific encoders (customize based on your sensor types)
        self.imu_encoder = nn.Sequential(
            nn.Linear(sensor_channels.get('imu', 6), embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(sensor_channels.get('camera', 3), 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embedding_dim // 4)  # Assuming 64x64 input images
        )
        
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(sensor_channels.get('depth', 1), 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, embedding_dim // 4)  # Assuming 64x64 input depth maps
        )
        
        self.position_encoder = nn.Sequential(
            nn.Linear(sensor_channels.get('position', 3), embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, sensor_data):
        """
        Process multi-modal sensor data
        
        Args:
            sensor_data: dictionary containing different sensor inputs
                - imu: [batch_size, imu_channels]
                - camera: [batch_size, camera_channels, height, width]
                - depth: [batch_size, depth_channels, height, width]
                - position: [batch_size, position_channels]
        """
        embeddings = []
        
        if 'imu' in sensor_data:
            imu_embedding = self.imu_encoder(sensor_data['imu'])
            embeddings.append(imu_embedding)
            
        if 'camera' in sensor_data:
            camera_embedding = self.camera_encoder(sensor_data['camera'])
            embeddings.append(camera_embedding)
            
        if 'depth' in sensor_data:
            depth_embedding = self.depth_encoder(sensor_data['depth'])
            embeddings.append(depth_embedding)
            
        if 'position' in sensor_data:
            position_embedding = self.position_encoder(sensor_data['position'])
            embeddings.append(position_embedding)
        
        # Concatenate all available sensor embeddings
        x = torch.cat(embeddings, dim=-1)
        
        # If we don't have all sensor types, we need to handle the dimension mismatch
        if x.shape[-1] < self.embedding_dim:
            # Pad with zeros as needed
            padded = torch.zeros(x.shape[0], self.embedding_dim, device=x.device)
            padded[:, :x.shape[-1]] = x
            x = padded
        
        # Pass through fusion layer
        x = self.fusion(x)
        return x

class UNetBlock(nn.Module):
    """Basic block with time conditioning for 1D state vectors."""
    def __init__(self, in_channels, out_channels, time_dim, use_attention=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.act1 = nn.SiLU()
        
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
            
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(out_channels, 4, batch_first=True)
            
    def forward(self, x, t):
        # Shortcut connection
        h = self.shortcut(x)
        
        # First block
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        # Add time conditioning
        time_emb = self.time_mlp(t)
        x = x + time_emb
        
        # Second block
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        # Attention if enabled
        if self.use_attention:
            # Reshape for attention [batch_size, 1, channels]
            x_attn = x.unsqueeze(1)
            x_attn, _ = self.attention(x_attn, x_attn, x_attn)
            x = x_attn.squeeze(1)
        
        # Add skip connection
        x = x + h
        return x

class DiffusionStateEstimator(nn.Module):
    """
    Diffusion model for UAV state estimation.
    Uses sensor data to predict clean UAV state from noisy observations.
    """
    def __init__(
        self, 
        state_dim=12,  # Position (3), Orientation (3), Linear Velocity (3), Angular Velocity (3)
        time_embedding_dim=128,
        sensor_channels={'imu': 6, 'camera': 3, 'depth': 1, 'position': 3},
        sensor_embedding_dim=128,
        hidden_dims=[128, 256, 512, 256, 128],
        use_attention=True
    ):
        super().__init__()
        self.state_dim = state_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_embedding_dim)
        
        # Sensor embedding
        self.sensor_embedding = SensorEmbedding(sensor_channels, sensor_embedding_dim)
        
        # Initial projection of noisy state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # Middle blocks
        self.blocks = nn.ModuleList()
        in_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            self.blocks.append(UNetBlock(in_dim, dim, time_embedding_dim, use_attention=(dim == max(hidden_dims) and use_attention)))
            in_dim = dim
            
        # Final prediction layer
        self.final = nn.Sequential(
            nn.Linear(in_dim, state_dim),
            nn.Tanh()  # Bound outputs to a reasonable range
        )
        
    def forward(self, noisy_state, timestep, sensor_data):
        """
        Args:
            noisy_state: [batch_size, state_dim] - Noisy state observation
            timestep: [batch_size] - Current diffusion timestep
            sensor_data: Dict of sensor readings
        """
        # Get embeddings
        t_emb = self.time_embedding(timestep)
        sensor_emb = self.sensor_embedding(sensor_data)
        
        # Encode state
        x = self.state_encoder(noisy_state)
        
        # Add sensor information
        x = x + sensor_emb
        
        # Process through UNet blocks
        for block in self.blocks:
            x = block(x, t_emb)
        
        # Get final prediction (this predicts the noise to subtract)
        noise_pred = self.final(x)
        
        return noise_pred 