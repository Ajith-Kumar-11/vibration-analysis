# 512x512 px images, 7 classes

import glob
import os
import pathlib
import sys
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


# Hyperparameters
NUM_CLASSES: int = 7  # Not read from config because of op12 and or3 subsets


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

  # Load datasets
  i_cwru: int = 0  # Hardcoded index of CWRU dataset in config
  train_path: str = os.path.join(config.datasets[i_cwru].fft_split_path, "train")
  test_path: str = os.path.join(config.datasets[i_cwru].fft_split_path, "test")
  train = DataLoader(torchvision.datasets.ImageFolder(train_path, transform=transformer), batch_size=64, shuffle=True)
  test = DataLoader(torchvision.datasets.ImageFolder(test_path, transform=transformer), batch_size=32, shuffle=True)

  # Log categories
  classes: list[str] = sorted([j.name.split("/")[-1] for j in pathlib.Path(train_path).iterdir()])
  logger.info(f"Found {len(classes)} categories in CWRU FFT dataset: {classes}")
  if len(classes) != NUM_CLASSES:
    logger.error(f"Expected {NUM_CLASSES} categories, got {len(classes)}")
    sys.exit()

  # CNN model
  class ConvNet(nn.Module):
    # CNN network
    def __init__(self, num_classes=NUM_CLASSES):
      super(ConvNet, self).__init__()
      # TODO
      pass

    # Feed forward network
    def forward(self, input):
      # TODO
      output = None
      # output=output.view(-1,dim*dim*dim)
      output = self.fc(output)
      return output
