from hashlib import sha256

from mafia.engine.seeds import game_seed, rng_for


def test_game_seed_pinned():
    assert game_seed(1, 2) == game_seed(1, 2) != game_seed(1, 3)
    assert game_seed(1, 2) == int.from_bytes(sha256(b"1:2").digest()[:8], "big")
    assert game_seed(1, 2) == 7438520176602755083
    assert rng_for(1, 2).random() == rng_for(1, 2).random()
