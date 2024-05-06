# 512x512 px images, 7 classes

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
from utility.file import count_files_with_extension

# Hyperparameters
NUM_CLASSES: int = 7  # Not read from config because of op12 and or3 subsets
BATCH_SIZE: int = 64
LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 0.0001
NUM_EPOCHS: int = 100


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
  train = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer), batch_size=BATCH_SIZE, shuffle=True
  )
  test = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer), batch_size=BATCH_SIZE, shuffle=True
  )
  train_count: int = count_files_with_extension(train_path, "png")
  test_count: int = count_files_with_extension(test_path, "png")
  logger.info(f"Loaded {train_count} images for training and {test_count} images for testing from CWRU FFT dataset")

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

  # Initialize
  model: ConvNet = ConvNet(num_classes=NUM_CLASSES).to(device)
  optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  loss_function = nn.CrossEntropyLoss()

  best_accuracy = 0.0

  for epoch in range(NUM_EPOCHS):
    model.train()
    train_accuracy: float = 0.0  # Reset
    train_loss: float = 0.0  # Reset

    # Training
    for i, (images, labels) in enumerate(train):
      if torch.cuda.is_available():
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
      optimizer.zero_grad()
      outputs = model(images)
      loss = loss_function(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.cpu().data * images.size(0)
      _, prediction = torch.max(outputs.data, 1)
      train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy: float = train_accuracy / train_count
    train_loss: float = train_loss / train_count

    # Testing
    model.eval()
    test_accuracy: float = 0.0
    for i, (images, labels) in enumerate(test):
      if torch.cuda.is_available():
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
      outputs = model(images)
      _, prediction = torch.max(outputs.data, 1)
      test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy: float = test_accuracy / test_count
    logger.info(f"Epoch: {epoch} Train Loss: {train_loss} Train: {train_accuracy}% Test: {test_accuracy}%")

    # Save/update the best model
    if test_accuracy > best_accuracy:
      torch.save(model.state_dict(), "best_checkpoint.model")  # TODO: Get path from config or set output folder
      best_accuracy = test_accuracy
