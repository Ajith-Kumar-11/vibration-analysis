import matplotlib.pyplot as plt
import numpy as np


def generate_spectrogram(
  data,  # Timeseries
  fs: int,  # Sampling frequency
  window_size: int = 512,
  overlap: int = 256,
):
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
  return (frequencies, times, spectrogram)


def save(width, height, location, frequencies, times, spectrogram):
  plt.figure(figsize=(width, height))
  plt.subplots_adjust(0, 0, 1, 1)
  plt.pcolormesh(times, frequencies, spectrogram.T, shading="auto")
  plt.axis("off")
  plt.savefig(location)
  plt.close(plt.gcf())
