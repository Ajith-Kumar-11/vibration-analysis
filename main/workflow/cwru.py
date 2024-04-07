import os
import sys
import fft.fft as fft
import numpy as np
import preprocess.preprocess as pre
import read.dataset.cwru
from config.config import Config
from loguru import logger


def generate_spectrograms(dfs: tuple[list, list], config: Config):
  icwru = 0  # Hardcoded index of the CWRU dataset in config file

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
    series = []  # Select FE sensor data for fan fault and DE sensor data for drive end fault
    if filename.startswith("fan"):
      series = df["FE"]
    elif filename.startswith("drive"):
      series = df["DE"]
    else:
      logger.error(f"Expected 'fan' or 'drive', got '{filename}'")
      sys.exit()
    chunks = pre.create_sublists(series, config.datasets[icwru].bucket_size)
    for i, chunk in enumerate(chunks):
      i_filename = f"{filename}{i}.png"
      location = os.path.join(config.datasets[icwru].fft_path, "faulty", i_filename)
      (frequencies, times, spectrogram) = fft.generate_spectrogram(chunk, config.datasets[icwru].frequency)
      fft.save(config.spectrogram.width, config.spectrogram.height, location, frequencies, times, spectrogram)


def _generate_location_metadata_normal(df):
  hp = df.at[0, "HP"]
  rpm = df.at[0, "RPM"]
  return f"{hp}hp-{rpm}rpm-"


def _generate_location_metadata_faulty(df):
  fault = read.dataset.cwru.decode_fault_type(df.at[0, "Location"])
  diameter = read.dataset.cwru.humanize_fault_diameter(df.at[0, "Fault"])
  hp = df.at[0, "HP"]
  rpm = df.at[0, "RPM"]
  return f"{fault}-{diameter}inch-{hp}hp-{rpm}rpm-"
