"""
This module provides a function to ensure that a folder exists at a specified path.

Functions:
    ensure_folder_exists(path: str) -> None: Ensures that a folder exists at the specified path. If it doesn't, it creates the folder.

Dependencies:
    os: For handling filesystem operations.
"""

import os


def ensure_folder_exists(path: str) -> None:
  """
  Ensures that a folder exists at the specified path. If it doesn't, it creates the folder.

  Args:
      path (str): The path to the folder to be ensured.

  Returns:
      None
  """
  if not os.path.exists(path):
    os.makedirs(path)
