import os


def ensure_folder_exists(path) -> None:
  if not os.path.exists(path):
    os.makedirs(path)
