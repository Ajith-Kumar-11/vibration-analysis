import os
import re
import sys
import pandas as pd
from scipy.io import loadmat
from utility import find_mat_files, flatten


def convert_mat_to_csv(filepath, metadata_sensor_location):
  # Get filename from full path
  filename = os.path.basename(filepath)
  filename, extension = os.path.splitext(filename)  # Strip extension from filename | Discard extension

  match_fault = re.search(r"(\d+)(inch)", filename)
  if not match_fault:
    print("Could not match fault diameter.\nTerminating program to prevent error...")
    sys.exit()
  metadata_fault = int(match_fault.group(1)) / 1000

  match_hp = re.search(r"(\d+)(hp)", filename)
  if not match_hp:
    print("Could not match motor HP.\nTerminating program to prevent error...")
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
        print("Could not match column ID.\nTerminating program to prevent error...")
        sys.exit()
      id = int(match_id.group())

  # Import columns
  ba = flatten(data[f"X{id}_BA_time"])
  de = flatten(data[f"X{id}_DE_time"])
  fe = flatten(data[f"X{id}_FE_time"])
  metadata_rpm = data[f"X{id}RPM"][0][0]

  # Append metadata
  length = len(de)
  location = [metadata_sensor_location] * length
  fault = [metadata_fault] * length  # Fault diameter
  hp = [metadata_hp] * length
  rpm = [metadata_rpm] * length

  # Create table
  df = pd.DataFrame(
    {
      "Location": location,
      "Fault": fault,
      "HP": hp,
      "RPM": rpm,
      "BA": ba,
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
metadata_sensor_location = "F-OR-Or3"
for mat_file in mat_files:
  convert_mat_to_csv(mat_file, metadata_sensor_location)
