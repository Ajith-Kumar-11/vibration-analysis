import os
import sys
import fft.fft as fft
import numpy as np
import polars as pl
import preprocess.preprocess as pre
import read.dataset.cwru
import utility.folder
from config.config import Config
from loguru import logger


def generate_spectrograms(dfs: tuple[list[pl.DataFrame], list[pl.DataFrame]], config: Config) -> None:
  icwru = 0  # Hardcoded index of the CWRU dataset in config file
  # Hardcoded names of output folders
  NORMAL: str = "normal"
  FAULTY: str = "faulty"

  utility.folder.ensure_folder_exists(config.datasets[icwru].fft_path)
  utility.folder.ensure_folder_exists(os.path.join(config.datasets[icwru].fft_path, NORMAL))
  utility.folder.ensure_folder_exists(os.path.join(config.datasets[icwru].fft_path, FAULTY))

  # Unpack as list of DFs
  normal: list[pl.DataFrame]
  faulty: list[pl.DataFrame]
  (normal, faulty) = dfs

  length_normal: int = len(normal)
  for idf, df in enumerate(normal):
    logger.info(f"Working on {idf + 1} of {length_normal} in normal CWRU dataset")
    filename = _generate_location_metadata_normal(df)
    series = np.concatenate((df["FE"], df["DE"]))
    chunks = pre.create_sublists(series, config.datasets[icwru].bucket_size)
    for i, chunk in enumerate(chunks):
      i_filename = f"{filename}{i}.png"
      location = os.path.join(config.datasets[icwru].fft_path, NORMAL, i_filename)
      (frequencies, times, spectrogram) = fft.generate_spectrogram(chunk, config.datasets[icwru].frequency)
      fft.save(config.spectrogram.width, config.spectrogram.height, location, frequencies, times, spectrogram)

  length_faulty: int = len(faulty)
  for idf, df in enumerate(faulty):
    logger.info(f"Working on {idf + 1} of {length_faulty} in faulty CWRU dataset")
    filename = _generate_location_metadata_faulty(df)
    series = _select_df_by_fault_location(df, filename)  # Fan fault => df[FE], drive fault => df[DE]
    chunks = pre.create_sublists(series, config.datasets[icwru].bucket_size)
    for i, chunk in enumerate(chunks):
      i_filename = f"{filename}{i}.png"
      location = os.path.join(config.datasets[icwru].fft_path, FAULTY, i_filename)
      (frequencies, times, spectrogram) = fft.generate_spectrogram(chunk, config.datasets[icwru].frequency)
      fft.save(config.spectrogram.width, config.spectrogram.height, location, frequencies, times, spectrogram)


def _generate_location_metadata_normal(df: pl.DataFrame) -> str:
  hp = df["HP"][0]
  return f"{hp}hp-"


def _generate_location_metadata_faulty(df: pl.DataFrame) -> str:
  fault = read.dataset.cwru.decode_fault_type(df["Location"][0])
  diameter = read.dataset.cwru.humanize_fault_diameter(df["Fault"][0])
  hp = df["HP"][0]
  return f"{fault}-{diameter}inch-{hp}hp-"


# Select FE sensor data for fan fault and DE sensor data for drive end fault
def _select_df_by_fault_location(df: pl.DataFrame, filename: str):
  if filename.startswith("fan"):
    return df["FE"]

  if filename.startswith("drive"):
    return df["DE"]

  logger.error(f"Expected 'fan' or 'drive', got '{filename}'")
  sys.exit()
