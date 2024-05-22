import config.config as cfg
import read.dataset.cwru
import workflow.cwru
import cnn.cnn as cnn


def main() -> None:
  config: cfg.Config = cfg.parse()

  # Generate spectrograms from CWRU dataset
  if False:
    cwru_dfs = read.dataset.cwru.read_from_folder(config)
    workflow.cwru.generate_spectrograms(cwru_dfs, config)

  # Train CNN model on CWRU FFT spectrograms
  if True:
    cnn.run(config)


if __name__ == "__main__":
  main()
