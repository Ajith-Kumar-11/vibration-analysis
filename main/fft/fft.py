"""
This module provides functions for generating and saving spectrograms from timeseries data.

Functions:
    generate_spectrogram(data, fs: int, window_size: int = 512, overlap: int = 256) -> tuple: Generates a spectrogram from the provided timeseries data.
    save(width: int, height: int, location: str, frequencies: np.ndarray, times: np.ndarray, spectrogram: np.ndarray) -> None: Saves the spectrogram as an image file.

Dependencies:
    matplotlib.pyplot: For plotting and saving the spectrogram.
    numpy: For numerical operations and FFT computations.
"""

import matplotlib.pyplot as plt
import numpy as np


def generate_spectrogram(
  data,  # Timeseries
  fs: int,  # Sampling frequency
  window_size: int = 512,
  overlap: int = 256,
) -> tuple:
  """
  Generates a spectrogram from the provided timeseries data.

  Args:
      data (numpy.ndarray): The timeseries data to generate the spectrogram from.
      fs (int): The sampling frequency of the timeseries data.
      window_size (int): The size of the window for FFT computation. Default is 512.
      overlap (int): The number of points of overlap between windows. Default is 256.

  Returns:
      tuple: A tuple containing the frequencies, times, and spectrogram array.
  """
  # Calculate the number of time points in the spectrogram
  num_time_points = int(np.floor((len(data) - window_size) / (window_size - overlap))) + 1

  # Initialize spectrogram array
  spectrogram = np.zeros((num_time_points, int(window_size / 2)))

  # Generate spectrogram
  for i in range(num_time_points):
    start = i * (window_size - overlap)
    end = start + window_size
    segment = data[start:end]

    # Apply windowing function
    segment *= np.hamming(window_size)

    # Compute FFT
    fft_result = np.fft.fft(segment)[: int(window_size / 2)]
    spectrogram[i, :] = np.abs(fft_result)

  # Return spectrogram
  times = np.arange(num_time_points) * (window_size - overlap) / fs
  frequencies = np.fft.fftfreq(int(window_size), 1 / fs)[: int(window_size / 2)]
  return frequencies, times, spectrogram


def save(
  width: int,
  height: int,
  location: str,
  frequencies: np.ndarray,
  times: np.ndarray,
  spectrogram: np.ndarray,
) -> None:
  """
  Saves the spectrogram as an image file.

  Args:
      width (int): The width of the saved image.
      height (int): The height of the saved image.
      location (str): The file path where the image will be saved.
      frequencies (numpy.ndarray): The array of frequencies.
      times (numpy.ndarray): The array of times.
      spectrogram (numpy.ndarray): The spectrogram array to be saved.

  Returns:
      None
  """
  plt.figure(figsize=(width, height))
  plt.subplots_adjust(0, 0, 1, 1)
  plt.pcolormesh(times, frequencies, spectrogram.T, shading="auto")
  plt.axis("off")
  plt.savefig(location, dpi=1)
  plt.close(plt.gcf())
