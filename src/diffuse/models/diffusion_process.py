import torch
import numpy as np
from tqdm import tqdm

class DiffusionProcess:
    """
    Implements the diffusion process for UAV state estimation.
    """
    def __init__(
        self,
        model,
        n_timesteps=1000,
        beta_schedule='linear',
        beta_start=1e-4,
        beta_end=0.02,
    ):
        """
        Args:
            model: The noise prediction model
            n_timesteps: Number of diffusion steps
            beta_schedule: Schedule for noise variance ('linear', 'cosine', or 'quadratic')
            beta_start: Starting noise value
            beta_end: Ending noise value
        """
        self.model = model
        self.n_timesteps = n_timesteps
        
        # Set up noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif beta_schedule == 'cosine':
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = n_timesteps + 1
            x = torch.linspace(0, n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / n_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        elif beta_schedule == 'quadratic':
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_timesteps) ** 2
            
        # Pre-compute values for diffusion and reverse process
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod) * x_0, (1 - alpha_cumprod) * I)
        
        Args:
            x_0: Clean state [B, state_dim]
            t: Timestep [B]
            noise: Optional pre-generated noise [B, state_dim]
            
        Returns:
            x_t: Noisy state at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Extract the appropriate timestep parameters
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Add noise according to the forward SDE
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def _extract(self, a, t, target_shape):
        """
        Extract values from tensor 'a' at indices 't' and reshape to target_shape
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        
        return out.reshape(batch_size, *((1,) * (len(target_shape) - 1))).to(t.device)
        
    def reverse_step(self, x_t, t, sensor_data):
        """
        Single step of the reverse diffusion process p(x_{t-1} | x_t)
        
        Args:
            x_t: Noisy state at timestep t [B, state_dim]
            t: Current timestep [B]
            sensor_data: Dictionary of sensor readings
        
        Returns:
            x_{t-1}: Predicted less noisy state
        """
        # Model predicts the noise component
        predicted_noise = self.model(x_t, t, sensor_data)
        
        # Get alpha and beta values for timestep t
        alpha_t = self._extract(self.alphas, t, x_t.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        
        # Calculate the coefficients for the mean of p(x_{t-1} | x_t, x_0)
        # We can predict x_0 as (x_t - sqrt(1-alpha_cumprod_t) * noise) / sqrt(alpha_cumprod_t)
        predicted_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Compute mean for p(x_{t-1} | x_t, x_0)
        mean = predicted_x0 * torch.sqrt(alpha_cumprod_t / alpha_t) + \
               x_t * (1 - alpha_cumprod_t/alpha_t)
        
        # If t = 0, we don't add noise
        if t[0] == 0:
            return mean
            
        # Compute variance
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        
        # Sample from the distribution
        noise = torch.randn_like(x_t)
        x_t_minus_1 = mean + torch.sqrt(variance) * noise
        
        return x_t_minus_1
        
    def sample(self, initial_state, sensor_data, device='cpu', show_progress=True):
        """
        Sample from the diffusion process to get a clean state estimate
        
        Args:
            initial_state: Initial noisy state estimate [B, state_dim]
            sensor_data: Dictionary of sensor readings
            device: Device to run sampling on
            show_progress: Whether to show a progress bar
            
        Returns:
            Predicted clean state [B, state_dim]
        """
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn_like(initial_state).to(device)
        
        # Iterate through the diffusion process in reverse
        iterator = tqdm(range(self.n_timesteps - 1, -1, -1)) if show_progress else range(self.n_timesteps - 1, -1, -1)
        
        for i in iterator:
            # Create timestep batch
            t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
            
            # Perform single reverse step
            with torch.no_grad():
                x = self.reverse_step(x, t, sensor_data)
                
        return x
    
    def physical_corrected_sample(self, initial_state, sensor_data, physics_model=None, device='cpu', show_progress=True):
        """
        Sample with physical constraints applied
        
        Args:
            initial_state: Initial noisy state [B, state_dim]
            sensor_data: Sensor readings
            physics_model: Optional UAV dynamics model to enforce physical constraints
            device: Device to run on
            show_progress: Whether to show progress bar
            
        Returns:
            Physically plausible state estimate [B, state_dim]
        """
        self.model.eval()
        
        # Get batch size and state dimension
        batch_size, state_dim = initial_state.shape
        
        # Start from pure noise
        x = torch.randn_like(initial_state).to(device)
        
        # Iterate through the diffusion process in reverse
        iterator = tqdm(range(self.n_timesteps - 1, -1, -1)) if show_progress else range(self.n_timesteps - 1, -1, -1)
        
        for i in iterator:
            # Create timestep batch
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Perform single reverse step
            with torch.no_grad():
                x = self.reverse_step(x, t, sensor_data)
                
                # Apply physical constraints if provided
                if physics_model is not None and i % 10 == 0:  # Apply every 10 steps to save computation
                    x = physics_model.enforce_constraints(x)
                    
        return x
    
    def train_step(self, clean_state, sensor_data, optimizer, device='cpu'):
        """
        Execute a single training step
        
        Args:
            clean_state: Ground truth UAV state [B, state_dim]
            sensor_data: Sensor readings
            optimizer: PyTorch optimizer
            device: Device to train on
            
        Returns:
            loss: The loss value for this batch
        """
        # Get batch size
        batch_size = clean_state.shape[0]
        
        # Move data to device
        clean_state = clean_state.to(device)
        for k, v in sensor_data.items():
            sensor_data[k] = v.to(device)
            
        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device).long()
        
        # Add noise to the clean state
        noisy_state, noise = self.add_noise(clean_state, t)
        
        # Predict the noise
        predicted_noise = self.model(noisy_state, t, sensor_data)
        
        # Compute loss (typically MSE between actual and predicted noise)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item() 