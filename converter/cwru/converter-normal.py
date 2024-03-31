import os
import re
import sys
import pandas as pd
from scipy.io import loadmat
from utility import find_mat_files, flatten


def convert_mat_to_csv(filepath):
  # Get filename from full path
  filename = os.path.basename(filepath)
  filename, extension = os.path.splitext(filename)  # Strip extension from filename | Discard extension

  match_hp = re.search(r"(\d+)(hp)", filename)
  if not match_hp:
    print("[ FATAL ] Could not match motor HP in `converter/cwru/converter-normal.py`")
    sys.exit()
  metadata_hp = int(match_hp.group(1))

  # Load MATLAB file
  data = loadmat(f"input/{filename}.mat")

  # Select the unique column ID | example X123RPM => 123
  id = 0
  for key in data.keys():
    if "RPM" in key:
      match_id = re.search(r"\d+", key)
      if not match_id:
        print("[ FATAL ] Could not match column ID in `converter/cwru/converter-normal.py`")
        sys.exit()
      id = match_id.group()

  # Special case (incorrectly formatted `normal-1hp` and `normal-2hp` from Case Western dataset)
  special_case_flag = False
  metadata_rpm = 0
  if id == 0:
    special_case_flag = True
    if filename == "normal-1hp":
      id = "098"
      metadata_rpm = 1772
    elif filename == "normal-2hp":
      id = "099"
      metadata_rpm = 1750
    else:
      print("[ FATAL ] Filename does not match any special case files in `converter/cwru/converter-normal.py`")
      sys.exit()

  # Import columns
  de = flatten(data[f"X{id}_DE_time"])
  fe = flatten(data[f"X{id}_FE_time"])
  if special_case_flag is False:
    metadata_rpm = data[f"X{id}RPM"][0][0]

  # Append metadata
  length = len(de)
  hp = [metadata_hp] * length
  rpm = [metadata_rpm] * length

  # Create table
  df = pd.DataFrame(
    {
      "HP": hp,
      "RPM": rpm,
      "DE": de,
      "FE": fe,
    }
  )

  # Preview and save
  print(df.head())
  if not os.path.exists("output/"):
    os.makedirs("output/")
  df.to_csv(f"output/{filename}.csv", index=False)


# Run script
mat_files = find_mat_files("input/")
for mat_file in mat_files:
  convert_mat_to_csv(mat_file)
