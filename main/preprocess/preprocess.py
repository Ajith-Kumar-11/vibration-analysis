def create_sublists(data, bucket_size):
  _range = range(0, len(data), bucket_size)
  return [data[i : i + bucket_size] for i in _range]
