import os
import pandas as pd


def as_list(directory) -> list[pd.DataFrame]:
  dfs: list[pd.DataFrame] = []

  # Iterate over each item in the directory
  for item in os.listdir(directory):
    item_path = os.path.join(directory, item)

    if os.path.isdir(item_path):
      # If the item is a directory, recursively call the function
      dfs.extend(as_list(item_path))
    elif item.endswith(".csv"):
      # If the item is a CSV file, read it into a DataFrame and append to the list
      df: pd.DataFrame = pd.read_csv(item_path)
      dfs.append(df)

  return dfs
