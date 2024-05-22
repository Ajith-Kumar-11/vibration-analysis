> [!WARNING]
> Following method is deprecated in favor of `random_split` from PyTorch utils. This section of code is preserved for documentation and should not be used.

# Split Folders
This is a utility to move FFT spectrograms into `train`, `val`, and `test` sets.

## Setup Environment
```sh
mamba create -n sf
mamba activate sf

# 3.10 is the latest version of python supported by `split-folders`
mamba install python=3.10
pip install split-folders # split-folders is not available on condaforge
```
