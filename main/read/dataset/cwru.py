import os
import sys
import read.csv
from loguru import logger


def read_from_folder(path):
  normal_folders: list[str] = ["12k Normal"]
  faulty_folders: list[str] = ["12k Fan End Bearing Fault", "12k Drive End Bearing Fault"]

  # Lists to store CSVs as DFs
  normal_dfs = []
  faulty_dfs = []

  for folder in normal_folders:
    folder_path = os.path.join(path, folder)
    df = read.csv.as_list(folder_path)
    normal_dfs.append(df)

  for folder in faulty_folders:
    folder_path = os.path.join(path, folder)
    df = read.csv.as_list(folder_path)
    faulty_dfs.append(df)

  logger.info(f"Read {len(normal_dfs)} normal and {len(faulty_dfs)} faulty CSVs from CWRU dataset")
  return (normal_dfs, faulty_dfs)


def decode_fault_type(fault_code):
  fault_code = fault_code.upper()
  main = _decode_main_fault(fault_code)
  secondary = _decode_secondary_fault(fault_code)
  if secondary == "outerrace":
    outerrace_fault = _decode_outerrace_fault(fault_code)
    secondary = f"{secondary}-{outerrace_fault}"
  return f"{main}-{secondary}"


def humanize_fault_diameter(value):
  humanized = str(value)
  humanized.replace("0", "").replace(".", "")
  return humanized


def _decode_main_fault(fault_code):
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


def _decode_secondary_fault(fault_code):
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
      logger.error(f"Expected 'I' or 'O' or 'B' at [{index}], got '{id}' from {_format(fault_code, index)}")
      sys.exit()


def _decode_outerrace_fault(fault_code):
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
      logger.error(f"Expected '6' or 'R' or 'P' at [{index}], got '{id}' from {_format(fault_code, index)}")
      sys.exit()


# Highlight selected index in string | (gamma, 2) => ga[m]ma
def _format(s: str, index: int) -> str:
  return f"{s[:index]}[{s[index]}]{s[index + 1:]}"
