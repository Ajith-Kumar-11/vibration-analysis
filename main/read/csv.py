"""
This module provides a function to recursively read all CSV files in a directory and its subdirectories into a list of Polars DataFrames.

Functions:
    as_list(directory: str) -> list[pl.DataFrame]: Recursively reads CSV files into a list of DataFrames.

Dependencies:
    os: For handling filesystem operations.
    polars: For reading CSV files into DataFrames.
"""

import os
import polars as pl


def as_list(directory: str) -> list[pl.DataFrame]:
  """
  Recursively reads all CSV files in a directory and its subdirectories into a list of Polars DataFrames.

  Args:
      directory (str): The path to the directory to search for CSV files.

  Returns:
      list[pl.DataFrame]: A list of DataFrames, each corresponding to a CSV file found in the directory and its subdirectories.
  """
  dfs: list[pl.DataFrame] = []

  # Iterate over each item in the directory
  for item in os.listdir(directory):
    item_path = os.path.join(directory, item)

    if os.path.isdir(item_path):
      # If the item is a directory, recursively call the function
      dfs.extend(as_list(item_path))
    elif item.endswith(".csv"):
      # If the item is a CSV file, read it into a DataFrame and append to the list
      df: pl.DataFrame = pl.read_csv(item_path)
      dfs.append(df)

  return dfs
