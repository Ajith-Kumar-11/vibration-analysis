import os
import sys
import numpy as np
import main.fft.fft as fft
import main.preprocess.preprocess as pre
import main.read.dataset.cwru as cwru

SAMPLING_FREQUENCY = 12000  # TODO: Fetch from config


def generate_spectrograms(dfs: tuple[list, list], bucket_size, image_size, save_folder):
  # Unpack as list of DFs
  (normal, faulty) = dfs

  for df in normal:
    filename = _generate_location_metadata_normal(df)
    series = np.concatenate((df["FE"], df["DE"]))
    chunks = pre.create_sublists(series, bucket_size)
    for i, chunk in enumerate(chunks):
      i_filename = f"{filename}{i}.png"
      location = os.path.join(save_folder, "normal", i_filename)
      (frequencies, times, spectrogram) = fft.generate_spectrogram(chunk, SAMPLING_FREQUENCY)
      fft.save_spectrogram(image_size, location, frequencies, times, spectrogram)

  for df in faulty:
    filename = _generate_location_metadata_faulty(df)
    series = []  # Select FE sensor data for fan fault and DE sensor data for drive end fault
    if filename.startswith("fan"):
      series = df["FE"]
    elif filename.startswith("drive"):
      series = df["DE"]
    else:
      print("Unexpected fault type in `main/workflow/cwru.py`. Terminating...")
      sys.exit()
    chunks = pre.create_sublists(series, bucket_size)
    for i, chunk in enumerate(chunks):
      i_filename = f"{filename}{i}.png"
      location = os.path.join(save_folder, "faulty", i_filename)
      (frequencies, times, spectrogram) = fft.generate_spectrogram(chunk, SAMPLING_FREQUENCY)
      fft.save_spectrogram(image_size, location, frequencies, times, spectrogram)


def _generate_location_metadata_normal(df):
  hp = df.at[0, "HP"]
  rpm = df.at[0, "RPM"]
  return f"{hp}hp-{rpm}rpm-"


def _generate_location_metadata_faulty(df):
  fault = cwru.decode_fault_type(df.at[0, "Location"])
  diameter = cwru.humanize_fault_diameter(df.at[0, "Fault"])
  hp = df.at[0, "HP"]
  rpm = df.at[0, "RPM"]
  return f"{fault}-{diameter}-{diameter}inch-{hp}hp-{rpm}rpm-"
