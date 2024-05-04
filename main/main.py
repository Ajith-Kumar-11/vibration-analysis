import config.config as cfg
import read.dataset.cwru
import workflow.cwru


def main() -> None:
  config: cfg.Config = cfg.parse()
  cwru_dfs = read.dataset.cwru.read_from_folder(config)
  workflow.cwru.generate_spectrograms(cwru_dfs, config)


if __name__ == "__main__":
  main()
