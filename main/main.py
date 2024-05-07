import config.config as cfg
import read.dataset.cwru
import workflow.cwru
import cnn.cwru.model_512_7
import cnn.cwru.model_512_19


def main() -> None:
  config: cfg.Config = cfg.parse()

  # Generate spectrograms from CWRU dataset
  if False:
    cwru_dfs = read.dataset.cwru.read_from_folder(config)
    workflow.cwru.generate_spectrograms(cwru_dfs, config)

  # Train CNN model on CWRU FFT spectrograms
  if True:
    cnn.cwru.model_512_7.run(config)
    cnn.cwru.model_512_19.run(config)


if __name__ == "__main__":
  main()
