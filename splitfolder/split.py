import splitfolders


ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)  # (train, val, test)
input_path: str = "input/path/here"
output_path: str = "input/path/here"
seed: int = 1337

splitfolders.ratio(input=input_path, output=output_path, seed=seed, ratio=ratio, group_prefix=None, move=False)
