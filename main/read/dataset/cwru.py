"""
This module provides functions to read datasets from folders, decode fault types, and humanize fault diameters.

Functions:
    read_from_folder(config: Config) -> tuple: Reads normal and faulty CSV files from specified folders in the configuration.
    decode_fault_type(fault_code: str) -> str: Decodes a fault code into a human-readable fault type string.
    humanize_fault_diameter(value: float) -> str: Converts a numeric fault diameter into a human-readable string.

Helper Functions:
    _decode_main_fault(fault_code: str) -> str: Decodes the main fault type from a fault code.
    _decode_secondary_fault(fault_code: str) -> str: Decodes the secondary fault type from a fault code.
    _decode_outerrace_fault(fault_code: str) -> str: Decodes the outer race fault type from a fault code.
    _format(s: str, index: int) -> str: Highlights the selected index in a string.

Dependencies:
    os: For handling filesystem operations.
    sys: For exiting the program in case of errors.
    polars: For handling DataFrame operations.
    read.csv: For reading CSV files as DataFrames.
    config.config: For accessing the configuration settings.
    loguru: For logging information and errors.
"""

import os
import sys
import polars as pl
import read.csv
from config.config import Config
from loguru import logger


def read_from_folder(config: Config) -> tuple:
  """
  Reads normal and faulty CSV files from specified folders in the configuration.

  Args:
      config (Config): The configuration object containing dataset paths.

  Returns:
      tuple: A tuple containing two lists of DataFrames: (normal_dfs, faulty_dfs).
  """
  icwru = 0  # Hardcoded index of the CWRU dataset in config file
  normal_folders: list[str] = ["12k Normal"]
  faulty_folders: list[str] = ["12k Fan End Bearing Fault", "12k Drive End Bearing Fault"]

  # Lists to store CSVs as DataFrames
  normal_dfs: list[pl.DataFrame] = []
  faulty_dfs: list[pl.DataFrame] = []

  for folder in normal_folders:
    folder_path = os.path.join(config.datasets[icwru].csv_path, folder)
    df = read.csv.as_list(folder_path)
    normal_dfs.extend(df)

  for folder in faulty_folders:
    folder_path = os.path.join(config.datasets[icwru].csv_path, folder)
    df = read.csv.as_list(folder_path)
    faulty_dfs.extend(df)

  logger.info(f"Read {len(normal_dfs)} normal and {len(faulty_dfs)} faulty CSVs from CWRU dataset")
  return normal_dfs, faulty_dfs


def decode_fault_type(fault_code: str) -> str:
  """
  Decodes a fault code into a human-readable fault type string.

  Args:
      fault_code (str): The fault code to decode.

  Returns:
      str: The decoded fault type string.
  """
  fault_code = fault_code.upper()
  main = _decode_main_fault(fault_code)
  secondary = _decode_secondary_fault(fault_code)
  if secondary == "outerrace":
    outerrace_fault = _decode_outerrace_fault(fault_code)
    secondary = f"{secondary}-{outerrace_fault}"
  return f"{main}-{secondary}"


def humanize_fault_diameter(value: float) -> str:
  """
  Converts a numeric fault diameter into a human-readable string.

  Args:
      value (float): The fault diameter value.

  Returns:
      str: The human-readable fault diameter string.
  """
  humanized: str = str(value)
  humanized = humanized.replace("0", "").replace(".", "")
  return humanized


def _decode_main_fault(fault_code: str) -> str:
  """
  Decodes the main fault type from a fault code.

  Args:
      fault_code (str): The fault code to decode.

  Returns:
      str: The main fault type string.
  """
  index = 0
  id = fault_code[index]
  match id:
    case "F":
      return "fan"
    case "D":
      return "drive"
    case _:
      logger.error(f"Expected 'F' or 'D' at [{index}], got '{id}' from {_format(fault_code, index)}")
      sys.exit()


def _decode_secondary_fault(fault_code: str) -> str:
  """
  Decodes the secondary fault type from a fault code.

  Args:
      fault_code (str): The fault code to decode.

  Returns:
      str: The secondary fault type string.
  """
  index = 2
  id = fault_code[index]
  match id:
    case "I":
      return "innerrace"
    case "O":
      return "outerrace"
    case "B":
      return "ball"
    case _:
      logger.error(f"Expected 'I', 'O', or 'B' at [{index}], got '{id}' from {_format(fault_code, index)}")
      sys.exit()


def _decode_outerrace_fault(fault_code: str) -> str:
  """
  Decodes the outer race fault type from a fault code.

  Args:
      fault_code (str): The fault code to decode.

  Returns:
      str: The outer race fault type string.
  """
  index = 6
  id = fault_code[index]
  match id:
    case "6":
      return "c6"
    case "R":
      return "or3"
    case "P":
      return "op12"
    case _:
      logger.error(f"Expected '6', 'R', or 'P' at [{index}], got '{id}' from {_format(fault_code, index)}")
      sys.exit()


def _format(s: str, index: int) -> str:
  """
  Highlights the selected index in a string.

  Args:
      s (str): The input string.
      index (int): The index to highlight.

  Returns:
      str: The formatted string with the selected index highlighted.
  """
  return f"{s[:index]}[{s[index]}]{s[index + 1:]}"
