# Vibration Analysis

## Setup Environment
```sh
# Create new virtual environment
mamba create -n vibration
mamba activate vibration

# Add python
mamba install python=3.12

# Add dependencies
mamba install numpy matplotlib

# Add dev dependencies
mamba install ruff loguru
```

## Notes
### CWRU Dataset
#### Normal FFT
- Normal dataset will produce twice the expected FFT images.
- This is because we consider both `df["FE]` (fan end) sensor data and `df["DE"]` (drive end) sensor data.
- In faulty data, only fan end or drive end sensor data is considered.
