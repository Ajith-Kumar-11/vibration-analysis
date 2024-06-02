"""
This module provides a function to count the number of files with a specific extension in a directory and its subdirectories.

Functions:
    count_files_with_extension(directory: str, extension: str) -> int: Counts the number of files with a specified extension.

Dependencies:
    os: For handling filesystem operations.
"""

import os


def count_files_with_extension(directory: str, extension: str) -> int:
  """
  Counts the number of files with a specific extension in a directory and its subdirectories.

  Args:
      directory (str): The path to the directory to search for files.
      extension (str): The extension of the files to count.

  Returns:
      int: The number of files with the specified extension.
  """
  extension = "." + extension.lower()
  count = 0
  for _, _, files in os.walk(directory):
    for file in files:
      if file.endswith(extension):
        count += 1
  return count
