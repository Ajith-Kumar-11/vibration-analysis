def create_sublists(data, bucket_size):
  """
  Splits the input data into sublists of a specified size.

  Args:
      data (list): The list of data to be split into sublists.
      bucket_size (int): The size of each sublist.

  Returns:
      list: A list of sublists, each of size `bucket_size`. The last sublist
            is dropped if it is not of the same size as the others.
  """
  _range = range(0, len(data), bucket_size)
  sublists = [data[i : i + bucket_size] for i in _range]

  # Drop last sublist if it's not the same size as the others
  if len(sublists[-1]) != bucket_size:
    sublists.pop()
  return sublists
