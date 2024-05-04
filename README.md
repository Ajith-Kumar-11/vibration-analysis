# Vibration Analysis

## Setup Environment
```sh
# Create new virtual environment
mamba create -n vibration
mamba activate vibration

# Add python
mamba install python=3.12

# Add dependencies
mamba install numpy matplotlib polars

# Add dev dependencies
mamba install ruff loguru

# Install PyTorch (if you have NVIDIA GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install PyTorch (for CPU; if you *don't* have NVIDIA GPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Notes
### CWRU Dataset
#### Normal FFT
- Normal dataset will produce twice the expected FFT images.
- This is because we consider both `df["FE"]` (fan end) sensor data and `df["DE"]` (drive end) sensor data.
- In faulty data, only fan end or drive end sensor data is considered.
