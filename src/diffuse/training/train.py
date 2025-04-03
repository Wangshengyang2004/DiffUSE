import torch
import numpy as np
import argparse
import os
import time
import sys
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from diffuse.models.diffusion_model import DiffusionStateEstimator
from diffuse.models.diffusion_process import DiffusionProcess
from diffuse.models.physics_model import UAVPhysicsModel
from diffuse.data.dataset import SimulatedUAVDataset

def train(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset
    train_dataset = SimulatedUAVDataset(
        data_dir=args.data_dir,
        split='train',
        sequence_length=args.sequence_length,
        sensor_types=args.sensor_types,
        domain_randomization=True
    )
    
    val_dataset = SimulatedUAVDataset(
        data_dir=args.data_dir,
        split='val',
        sequence_length=args.sequence_length,
        sensor_types=args.sensor_types,
        domain_randomization=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Calculate sensor channels based on dataset
    sample = train_dataset[0]
    sensor_data = sample['sensor_data']
    sensor_channels = {}
    
    for sensor_type, data in sensor_data.items():
        if len(data.shape) == 1:  # 1D data like IMU or position
            sensor_channels[sensor_type] = data.shape[0]
        elif len(data.shape) == 3:  # Image data (C, H, W)
            sensor_channels[sensor_type] = data.shape[0]
    
    # Initialize model
    model = DiffusionStateEstimator(
        state_dim=args.state_dim,
        time_embedding_dim=args.time_embedding_dim,
        sensor_channels=sensor_channels,
        sensor_embedding_dim=args.sensor_embedding_dim,
        hidden_dims=args.hidden_dims,
        use_attention=args.use_attention
    ).to(device)
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(
        model=model,
        n_timesteps=args.diffusion_steps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    
    # Initialize physics model for consistency checking
    physics_model = UAVPhysicsModel() if args.use_physics else None
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # Create scheduler if needed
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
    
    # Training loop
    best_val_loss = float('inf')
    step = 0
    
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_losses = []
        
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            # Get clean state and sensor data
            clean_state = batch['clean_state'].to(device)
            sensor_data = {k: v.to(device) for k, v in batch['sensor_data'].items()}
            
            # Train diffusion model
            loss = diffusion.train_step(clean_state, sensor_data, optimizer, device)
            train_losses.append(loss)
            
            # Log progress
            if batch_idx % args.log_interval == 0:
                print(f"Epoch: {epoch+1}/{args.epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss:.4f}")
            
            step += 1
        
        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f}s. Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Get clean state and sensor data
                clean_state = batch['clean_state'].to(device)
                sensor_data = {k: v.to(device) for k, v in batch['sensor_data'].items()}
                
                # Sample random timesteps
                batch_size = clean_state.shape[0]
                t = torch.randint(0, diffusion.n_timesteps, (batch_size,), device=device).long()
                
                # Add noise to the clean state
                noisy_state, noise = diffusion.add_noise(clean_state, t)
                
                # Predict the noise
                predicted_noise = model(noisy_state, t, sensor_data)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                val_losses.append(loss.item())
                
                # If using physics consistency, also calculate physics score
                if args.use_physics and batch_idx % 10 == 0:
                    # Get a sequence of states if available
                    if 'next_clean_state' in batch:
                        next_state = batch['next_clean_state'].to(device)
                        physics_score = physics_model.consistency_score(clean_state, next_state)
                        avg_physics_score = physics_score.mean().item()
                        print(f"Batch {batch_idx}, Physics consistency score: {avg_physics_score:.4f}")
        
        # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Update learning rate scheduler if needed
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }
            
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Model saved with validation loss: {best_val_loss:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }
            
            torch.save(checkpoint, os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model for UAV state estimation")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data/simulated', help='Directory containing dataset')
    parser.add_argument('--sensor_types', type=str, nargs='+', default=['imu', 'camera', 'depth', 'position'], 
                        help='Sensor types to use')
    parser.add_argument('--sequence_length', type=int, default=1, help='Length of state sequence')
    
    # Model parameters
    parser.add_argument('--state_dim', type=int, default=12, help='Dimension of UAV state')
    parser.add_argument('--time_embedding_dim', type=int, default=128, help='Dimension of time embedding')
    parser.add_argument('--sensor_embedding_dim', type=int, default=128, help='Dimension of sensor embedding')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256, 512, 256, 128], 
                        help='Hidden dimensions for UNet blocks')
    parser.add_argument('--use_attention', action='store_true', help='Use attention in UNet blocks')
    
    # Diffusion parameters
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine', 'quadratic'],
                        help='Schedule for noise variance')
    parser.add_argument('--beta_start', type=float, default=1e-4, help='Starting noise value')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Ending noise value')
    
    # Physics model parameters
    parser.add_argument('--use_physics', action='store_true', help='Use physics model for consistency')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval (batches)')
    parser.add_argument('--save_interval', type=int, default=5, help='Save interval (epochs)')
    
    args = parser.parse_args()
    train(args) 