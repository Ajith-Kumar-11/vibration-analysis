import os
import sys
import torch
import torch.nn as nn
from config.config import Config
from loguru import logger
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms
from utility.folder import ensure_folder_exists

# Hyperparameters
BATCH_SIZE: int = 1
LEARNING_RATE: float = 0.0001
WEIGHT_DECAY: float = 0.0001
NUM_EPOCHS: int = 10
SPLIT_RATIO: float = 0.5  # Train-test split


def run(config: Config, num_classes: int, subfolder: str) -> None:
  print(f"\nWorking on 512x512 px images, {num_classes} classes")

  # Generate folder to save model
  model_output_folder: str = f"output/model/modelx/{subfolder}"
  ensure_folder_exists(model_output_folder)

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

  # Load dataset
  i_cwru: int = 0  # Hardcoded index of CWRU dataset in config
  fft_classes_directory: str = os.path.join(config.datasets[i_cwru].fft_classes, subfolder)
  dataset = datasets.ImageFolder(root=fft_classes_directory, transform=transformer)

  # Split dataset
  train_size = int(0.5 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

  # Count images
  train_count: int = len(train_loader)
  test_count: int = len(test_loader)
  logger.info(f"Loaded {train_count} images for training and {test_count} images for testing from {subfolder} dataset")

  # Log categories
  classes: list[str] = sorted(dataset.classes)
  logger.info(f"Found {len(classes)} categories in {subfolder} dataset: {classes}")
  if len(classes) != num_classes:
    logger.error(f"Expected {num_classes} categories, got {len(classes)}")
    sys.exit()

  # CNN model
  class ConvNet(nn.Module):
    # CNN network
    def __init__(self, num_classes=num_classes):
      super(ConvNet, self).__init__()
      # (BatchSize, 3, 512, 512) -> (BatchSize, 16, 512, 512)
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
      self.bn1 = nn.BatchNorm2d(num_features=16)
      self.relu1 = nn.ReLU()

      # (BatchSize, 16, 512, 512) -> (BatchSize, 16, 256, 256)
      self.pool1 = nn.MaxPool2d(kernel_size=2)

      # (BatchSize, 16, 256, 256) -> (BatchSize, 32, 256, 256)
      self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
      self.bn2 = nn.BatchNorm2d(num_features=32)
      self.relu2 = nn.ReLU()

      # (BatchSize, 32, 256, 256) -> (BatchSize, 32, 128, 128)
      self.pool2 = nn.MaxPool2d(kernel_size=2)

      # (BatchSize, 32, 128, 128) -> (BatchSize, 64, 128, 128)
      self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
      self.bn3 = nn.BatchNorm2d(num_features=64)
      self.relu3 = nn.ReLU()

      # (BatchSize, 64, 128, 128) -> (BatchSize, 64, 64, 64)
      self.pool3 = nn.MaxPool2d(kernel_size=2)

      # Fully connected layer
      self.fc = nn.Linear(in_features=64 * 64 * 64, out_features=num_classes)

    # Feed forward network
    def forward(self, input):
      output = self.conv1(input)
      output = self.bn1(output)
      output = self.relu1(output)
      output = self.pool1(output)

      output = self.conv2(output)
      output = self.bn2(output)
      output = self.relu2(output)
      output = self.pool2(output)

      output = self.conv3(output)
      output = self.bn3(output)
      output = self.relu3(output)
      output = self.pool3(output)

      output = output.view(-1, 64 * 64 * 64)  # Flatten
      output = self.fc(output)
      return output

  # Initialize
  model: ConvNet = ConvNet(num_classes=num_classes).to(device)
  logger.info(f"Initialized {subfolder} CNN model")
  optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  loss_function = nn.CrossEntropyLoss()

  best_accuracy = 0.0

  for epoch in range(NUM_EPOCHS):
    logger.info(f"Working on epoch {epoch + 1} of {NUM_EPOCHS}")
    model.train()
    train_accuracy: float = 0.0  # Reset
    train_loss: float = 0.0  # Reset

    # Training
    for i, (images, labels) in enumerate(train_loader):
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
    for i, (images, labels) in enumerate(test_loader):
      if torch.cuda.is_available():
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
      outputs = model(images)
      _, prediction = torch.max(outputs.data, 1)
      test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy: float = test_accuracy / test_count
    logger.info(
      f"Train Loss: {train_loss} Train: {train_accuracy * 100 / BATCH_SIZE}% Test: {test_accuracy * 100 / BATCH_SIZE}%"
    )

    # Save/update the best model
    if test_accuracy > best_accuracy:
      torch.save(model.state_dict(), os.path.join(model_output_folder, "checkpoint.model"))
      best_accuracy = test_accuracy
