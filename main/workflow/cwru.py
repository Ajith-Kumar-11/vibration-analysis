import os
import sys
import fft.fft as fft
import numpy as np
import preprocess.preprocess as pre
import read.dataset.cwru
from config.config import Config
from loguru import logger
import utility.folder


def generate_spectrograms(dfs: tuple[list, list], config: Config) -> None:
  icwru = 0  # Hardcoded index of the CWRU dataset in config file
  utility.folder.ensure_folder_exists(config.datasets[icwru].fft_path)

  # Unpack as list of DFs
  (normal, faulty) = dfs

  for df in normal:
    filename = _generate_location_metadata_normal(df)
    series = np.concatenate((df["FE"], df["DE"]))
    chunks = pre.create_sublists(series, config.datasets[icwru].bucket_size)
    for i, chunk in enumerate(chunks):
      i_filename = f"{filename}{i}.png"
      location = os.path.join(config.datasets[icwru].fft_path, "normal", i_filename)
      (frequencies, times, spectrogram) = fft.generate_spectrogram(chunk, config.datasets[icwru].frequency)
      fft.save(config.spectrogram.width, config.spectrogram.height, location, frequencies, times, spectrogram)

  for df in faulty:
    filename = _generate_location_metadata_faulty(df)
    series = _select_df_by_fault_location(df, filename)  # Fan fault => df[FE], drive fault => df[DE]
    chunks = pre.create_sublists(series, config.datasets[icwru].bucket_size)
    for i, chunk in enumerate(chunks):
      i_filename = f"{filename}{i}.png"
      location = os.path.join(config.datasets[icwru].fft_path, "faulty", i_filename)
      (frequencies, times, spectrogram) = fft.generate_spectrogram(chunk, config.datasets[icwru].frequency)
      fft.save(config.spectrogram.width, config.spectrogram.height, location, frequencies, times, spectrogram)


def _generate_location_metadata_normal(df) -> str:
  hp = df.at[0, "HP"]
  rpm = df.at[0, "RPM"]
  return f"{hp}hp-{rpm}rpm-"


def _generate_location_metadata_faulty(df) -> str:
  fault = read.dataset.cwru.decode_fault_type(df.at[0, "Location"])
  diameter = read.dataset.cwru.humanize_fault_diameter(df.at[0, "Fault"])
  hp = df.at[0, "HP"]
  rpm = df.at[0, "RPM"]
  return f"{fault}-{diameter}inch-{hp}hp-{rpm}rpm-"


# Select FE sensor data for fan fault and DE sensor data for drive end fault
def _select_df_by_fault_location(df, filename: str):
  if filename.startswith("fan"):
    return df["FE"]

  if filename.startswith("drive"):
    return df["DE"]

  logger.error(f"Expected 'fan' or 'drive', got '{filename}'")
  sys.exit()
