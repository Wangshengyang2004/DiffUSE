# DiffUSE: Diffusion-Based State Estimation for Agile UAVs

DiffUSE (Diffusion-Based UAV State Estimation) is a framework for cost-effective, robust sim-to-real transfer in agile unmanned aerial vehicles (UAVs). The framework leverages conditional diffusion models for UAV state estimation using low-cost onboard sensors.

## Key Features

- **Generative Multi-Modal Filtering**: Reframes UAV state estimation as a conditional generative denoising process that integrates multi-modal sensor data
- **Weakly-Supervised Sim-to-Real Adaptation**: Implements extensive domain randomization within simulation environments
- **Real-Time Inference Enhanced by Physical Priors**: Incorporates known UAV physical dynamics and kinematic constraints

## Project Structure

```
DiffUSE/
├── src/
│   └── diffuse/
│       ├── models/
│       │   ├── diffusion_model.py  - Core diffusion model implementation
│       │   ├── diffusion_process.py - Diffusion process implementation
│       │   └── physics_model.py - UAV physics model for constraints
│       ├── data/
│       │   └── dataset.py - Dataset implementation for UAV data
│       ├── training/
│       │   └── train.py - Training script
│       ├── inference/
│       │   └── predict.py - Inference script
│       └── sim/
│           └── generate_data.py - Simulation data generation
├── data/
│   └── simulated/ - Simulated data directory
├── checkpoints/ - Model checkpoints directory
└── requirements.txt - Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DiffUSE.git
cd DiffUSE
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For development, install with development dependencies:
```bash
pip install -e ".[dev]"
```

## Development

### Running Tests

Run the unit tests:

```bash
# Run all tests
python tests/run_tests.py

# Run with coverage report
python tests/run_coverage.py
```

### Code Style

Format code according to project style:

```bash
# Format code with black
black src tests

# Sort imports
isort src tests

# Run linting
flake8 src tests
```

## Usage

### Generating Simulated Data

Generate simulated UAV flight data for training:

```bash
python src/diffuse/sim/generate_data.py --output_dir data/simulated --num_trajectories 100
```

Options:
- `--trajectory_type`: Type of trajectory ('random', 'circle', 'figure8', 'hover')
- `--num_trajectories`: Number of trajectories to generate
- `--trajectory_duration`: Duration of each trajectory in seconds

### Training

Train the diffusion model using simulated data:

```bash
python src/diffuse/training/train.py --data_dir data/simulated --output_dir checkpoints
```

Options:
- `--data_dir`: Directory containing dataset
- `--sensor_types`: Sensor types to use (e.g., imu camera depth position)
- `--use_physics`: Use physics model for consistency
- `--diffusion_steps`: Number of diffusion steps
- `--epochs`: Number of training epochs

### Inference

Run inference with a trained model:

```bash
python src/diffuse/inference/predict.py --model_path checkpoints/best_model.pth --imu_file data.npy
```

Options:
- `--model_path`: Path to model checkpoint
- `--imu_file`, `--camera_file`, `--depth_file`, `--position_file`: Paths to sensor data
- `--use_physics`: Use physics model for constraints
- `--diffusion_steps`: Number of diffusion steps for inference

## How It Works

DiffUSE works as follows:

1. **Data Collection**: Multi-modal sensor data is collected from the UAV (IMU, camera, depth, position)
2. **Noise Addition**: Initial rough state estimates are obtained from sensor fusion
3. **Diffusion Process**: The diffusion model progressively refines the noisy state estimate
4. **Physical Constraints**: Physical priors are enforced to ensure plausible state predictions

The diffusion model is trained to predict the clean state from noisy observations by learning the reverse diffusion process. At inference time, the model iteratively denoises the state estimate starting from random noise.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{DiffUSE2023,
  title={DiffUSE: Diffusion-Based State Estimation for Agile UAVs with Low-Cost Sim-to-Real Transfer Using Multi-Modal Sensors},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
