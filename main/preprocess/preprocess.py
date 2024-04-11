def create_sublists(data, bucket_size):
  _range = range(0, len(data), bucket_size)
  sublists = [data[i : i + bucket_size] for i in _range]

  # Drop last sublist if it's not the same size as the others
  if len(sublists[-1]) != bucket_size:
    sublists.pop()
  return sublists
