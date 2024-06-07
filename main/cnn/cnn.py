import cnn.cwru.model_x
from config.config import Config


def run(config: Config) -> None:
  # Runner for generalized CNN model (Model X)
  cnn.cwru.model_x.run(config, 2, "02A_512")
  cnn.cwru.model_x.run(config, 7, "07A_512")
  cnn.cwru.model_x.run(config, 7, "07B_512")
  cnn.cwru.model_x.run(config, 7, "07C_512")
  cnn.cwru.model_x.run(config, 7, "07D_512")
  cnn.cwru.model_x.run(config, 7, "07E_512")
  cnn.cwru.model_x.run(config, 19, "19A_512")
  cnn.cwru.model_x.run(config, 65, "66A_512")  # TODO: F-IR-21in-1hp is not present in FFT dataset
