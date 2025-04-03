"""
Tests for training module.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path

from src.diffuse.models.diffusion_model import DiffusionModel, DenoisingUNet
from src.diffuse.training.train import DiffusionTrainer


class MockDataset:
    """Mock dataset for training."""
    
    def __init__(self, num_samples=100, state_dim=12, condition_dim=64):
        self.num_samples = num_samples
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Return a dict with clean_state and sensor_data
        return {
            'clean_state': torch.randn(self.state_dim),
            'sensor_data': {
                'combined': torch.randn(self.condition_dim)
            }
        }


class TestDiffusionTrainer:
    
    @pytest.fixture
    def trainer_setup(self, tmp_path):
        """Set up a trainer with mock model and data."""
        # Create mock model
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
        
        # Create mock datasets
        train_dataset = MockDataset(num_samples=100)
        val_dataset = MockDataset(num_samples=20)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create trainer
        trainer = DiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path / "checkpoints",
            learning_rate=1e-4,
            weight_decay=1e-5,
            warmup_steps=10
        )
        
        return {
            'trainer': trainer,
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'output_dir': tmp_path / "checkpoints"
        }
    
    def test_init(self, trainer_setup):
        """Test trainer initialization."""
        trainer = trainer_setup['trainer']
        model = trainer_setup['model']
        
        assert trainer.model is model
        assert trainer.learning_rate == 1e-4
        assert trainer.weight_decay == 1e-5
        assert trainer.warmup_steps == 10
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LambdaLR)
        
    def test_save_checkpoint(self, trainer_setup):
        """Test saving checkpoints."""
        trainer = trainer_setup['trainer']
        output_dir = trainer_setup['output_dir']
        
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(0)
        
        # Check that checkpoint file exists
        assert os.path.exists(checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Check checkpoint contents
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'scheduler_state_dict' in checkpoint
        assert 'train_losses' in checkpoint
        assert 'val_losses' in checkpoint
        
    def test_train_step(self, trainer_setup):
        """Test a single training step."""
        trainer = trainer_setup['trainer']
        train_loader = trainer_setup['train_loader']
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Run training step
        loss = trainer.train_step(batch)
        
        # Check loss
        assert isinstance(loss, float)
        assert loss > 0
        
    def test_validation_step(self, trainer_setup):
        """Test a single validation step."""
        trainer = trainer_setup['trainer']
        val_loader = trainer_setup['val_loader']
        
        # Get a batch
        batch = next(iter(val_loader))
        
        # Run validation step
        loss = trainer.validation_step(batch)
        
        # Check loss
        assert isinstance(loss, float)
        assert loss > 0
        
    def test_warmup_scheduler(self, trainer_setup):
        """Test the warmup scheduler."""
        trainer = trainer_setup['trainer']
        
        # Initial learning rate
        initial_lr = trainer.scheduler.get_last_lr()[0]
        
        # Step through warmup
        for _ in range(trainer.warmup_steps):
            trainer.scheduler.step()
            
        # Learning rate after warmup should be higher
        warmup_lr = trainer.scheduler.get_last_lr()[0]
        assert warmup_lr > initial_lr
        
        # Continue stepping - should start decaying
        for _ in range(100):
            trainer.scheduler.step()
            
        # Final learning rate should be lower than peak
        final_lr = trainer.scheduler.get_last_lr()[0]
        assert final_lr < warmup_lr
        
    def test_train_epoch(self, trainer_setup):
        """Test training for a single epoch."""
        trainer = trainer_setup['trainer']
        
        # Train for one epoch
        train_loss = trainer.train_epoch()
        
        # Check loss
        assert isinstance(train_loss, float)
        assert train_loss > 0
        
        # Check that train_losses has been updated
        assert len(trainer.train_losses) == 1
        
    def test_validate(self, trainer_setup):
        """Test validation."""
        trainer = trainer_setup['trainer']
        
        # Run validation
        val_loss = trainer.validate()
        
        # Check loss
        assert isinstance(val_loss, float)
        assert val_loss > 0
        
        # Check that val_losses has been updated
        assert len(trainer.val_losses) == 1
        
    def test_load_checkpoint(self, trainer_setup):
        """Test loading a checkpoint."""
        trainer = trainer_setup['trainer']
        model = trainer_setup['model']
        
        # Save initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone()
        
        # Train for one step to change parameters
        batch = next(iter(trainer_setup['train_loader']))
        trainer.train_step(batch)
        
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(0)
        
        # Change parameters again
        batch = next(iter(trainer_setup['train_loader']))
        trainer.train_step(batch)
        
        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
        
        # Check that parameters are restored to values at checkpoint
        for name, param in model.named_parameters():
            # Parameters should be different from initial values but
            # should match the values at checkpoint save time
            assert not torch.allclose(param, initial_params[name])
            
    def test_fit(self, trainer_setup):
        """Test the full training loop for a small number of epochs."""
        trainer = trainer_setup['trainer']
        
        # Override model's forward pass to speed up testing
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = nn.Parameter(torch.zeros(1))
                
            def compute_loss(self, x_0, condition=None):
                return self.dummy_param.sum()
                
        trainer.model = MockModel()
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
        
        # Train for 2 epochs
        trainer.fit(num_epochs=2, validate_every=1)
        
        # Check that losses have been tracked
        assert len(trainer.train_losses) == 2
        assert len(trainer.val_losses) == 2 