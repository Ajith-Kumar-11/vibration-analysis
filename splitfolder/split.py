import splitfolders


ratio: tuple[float, float, float] = (0.5, 0.0, 0.5)  # (train, val, test)
input_path: str = "input/path/here"
output_path: str = "input/path/here"
seed: int = 1337

splitfolders.ratio(input=input_path, output=output_path, seed=seed, ratio=ratio, group_prefix=None, move=False)
