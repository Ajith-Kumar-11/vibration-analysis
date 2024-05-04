# 512x512 px images, 7 classes

import glob
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from config.config import Config
from loguru import logger
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def run(config: Config) -> None:
  # Prefer CUDA capable GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logger.info(f"PyTorch device: {device}")

  # Image transformer
  transformer = transforms.Compose(
    [
      transforms.Resize((config.spectrogram.width, config.spectrogram.height)),
      transforms.ToTensor(),  # Convert image to tensor
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale tensor values from [0, 1] to [-1, 1]
    ]
  )
