import random
from hashlib import sha256


def game_seed(run_seed: int, game_index: int) -> int:
    return int.from_bytes(sha256(f"{run_seed}:{game_index}".encode()).digest()[:8], "big")


def rng_for(run_seed: int, game_index: int) -> random.Random:
    return random.Random(game_seed(run_seed, game_index))
