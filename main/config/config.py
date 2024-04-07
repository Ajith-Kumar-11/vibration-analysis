import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Spectrogram:
  width: int
  height: int


@dataclass(frozen=True, slots=True)
class Dataset:
  id: int
  name: str
  csv_path: str
  fft_path: str
  frequency: int
  bucket_size: int


@dataclass(frozen=True, slots=True)
class Config:
  spectrogram: Spectrogram
  datasets: list[Dataset]


def parse(json_path: str = "config/config.json") -> Config:
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
        frequency=dataset["frequency"],
        bucket_size=dataset["bucket_size"],
      )
    )

  return Config(spectrogram, datasets)
