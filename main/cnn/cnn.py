import cnn.cwru.model_x
from config.config import Config


def run(config: Config) -> None:
  # Runner for generalized CNN model (Model X)
  cnn.cwru.model_x.run(config, 7, "512_7")
  cnn.cwru.model_x.run(config, 19, "512_19")
