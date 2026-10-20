import random
from hashlib import sha256


def game_seed(run_seed: int, game_index: int) -> int:
    return int.from_bytes(sha256(f"{run_seed}:{game_index}".encode()).digest()[:8], "big")


def rng_for(run_seed: int, game_index: int, stream: str = "") -> random.Random:
    """Independent RNG per purpose ("roles", "votes3", ...) so paired games share engine randomness."""
    if not stream:
        return random.Random(game_seed(run_seed, game_index))
    return random.Random(int.from_bytes(sha256(f"{run_seed}:{game_index}:{stream}".encode()).digest()[:8], "big"))
