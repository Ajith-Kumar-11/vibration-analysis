"""
This module provides classes and functions for handling the configuration of spectrograms and datasets
through JSON files.

Classes:
    Spectrogram: Represents the configuration of a spectrogram with width and height.
    Dataset: Represents the configuration details of a dataset.
    Config: Represents the complete configuration, including spectrogram and datasets.

Functions:
    parse(json_path: str = "config/config_local.json") -> Config: Parses the configuration from a JSON file.
    _log(config: Config) -> None: Logs the configuration details.

Dependencies:
    json: For handling JSON files.
    dataclasses: For defining data structures.
    loguru: For logging configuration details.
"""

import json
from dataclasses import dataclass
from loguru import logger


@dataclass(frozen=True, slots=True)
class Spectrogram:
  """
  Represents the configuration of a spectrogram.

  Attributes:
      width (int): The width of the spectrogram.
      height (int): The height of the spectrogram.
  """

  width: int
  height: int


@dataclass(frozen=True, slots=True)
class Dataset:
  """
  Represents the configuration details of a dataset.

  Attributes:
      id (int): The unique identifier for the dataset.
      name (str): The name of the dataset.
      csv_path (str): The path to the CSV file associated with the dataset.
      fft_path (str): The path to the FFT data file associated with the dataset.
      fft_classes (str): The classes of FFT data.
      frequency (int): The frequency parameter of the dataset.
      bucket_size (int): The bucket size parameter of the dataset.
  """

  id: int
  name: str
  csv_path: str
  fft_path: str
  fft_classes: str
  frequency: int
  bucket_size: int


@dataclass(frozen=True, slots=True)
class Config:
  """
  Represents the complete configuration, including spectrogram and datasets.

  Attributes:
      spectrogram (Spectrogram): The spectrogram configuration.
      datasets (list[Dataset]): A list of dataset configurations.
  """

  spectrogram: Spectrogram
  datasets: list[Dataset]


def parse(json_path: str = "config/config_local.json") -> Config:
  """
  Parses the configuration from a JSON file.

  Args:
      json_path (str): The path to the JSON configuration file. Default is "config/config_local.json".

  Returns:
      Config: The parsed configuration object.
  """
  with open(json_path, "r") as f:
    json_data = json.load(f)
  spectrogram_data = json_data["spectrogram"]
  spectrogram = Spectrogram(width=spectrogram_data["width"], height=spectrogram_data["height"])

  datasets = []
  for dataset in json_data["datasets"]:
    datasets.append(
      Dataset(
        id=dataset["id"],
        name=dataset["name"],
        csv_path=dataset["csv_path"],
        fft_path=dataset["fft_path"],
        fft_classes=dataset["fft_classes"],
        frequency=dataset["frequency"],
        bucket_size=dataset["bucket_size"],
      )
    )

  config = Config(spectrogram, datasets)
  _log(config)
  return config


def _log(config: Config) -> None:
  """
  Logs the configuration details.

  Args:
      config (Config): The configuration object to log.
  """
  log = []
  log.append("Read configuration successfully")
  log.append(config.spectrogram)
  for dataset in config.datasets:
    log.append(dataset)

  logger.info("\n".join([str(x) for x in log]))
