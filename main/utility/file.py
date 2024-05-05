import os


def count_files_with_extension(directory: str, extension: str) -> int:
  extension = "." + extension.lower()
  count = 0
  for _, _, files in os.walk(directory):
    for file in files:
      if file.endswith(extension):
        count += 1
  return count
