from functools import cache
from importlib.resources import files


@cache
def template(name: str) -> str:
    return files(__package__).joinpath(f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")
