import json
from dataclasses import dataclass


@dataclass
class MyDataClass:
  spectrogram: int
  datasets: dict


def parse(json_path):
  with open(json_path, "r") as f:
    json_dict = json.load(f)
  return MyDataClass(**json_dict)
