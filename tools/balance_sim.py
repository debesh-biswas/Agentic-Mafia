"""Random-play balance check used during design (2026-10-19).

Plays N games with players who act uniformly at random under the locked rules:
7 players (1 mafia, 1 doctor, 5 villagers), night start, doctor may not self-protect
two nights in a row, plurality vote with tie = no elimination, mafia wins at parity,
town wins if day 8 is reached.

Reference results (200,000 games): town 39.3%, mafia 60.7%, doctor saves 0.41/game,
median end day 3, P(reach the morning of day 8) ~ 1.8e-04 (1M games). The real engine's `mafia sim` must reproduce these.

Usage: python tools/balance_sim.py [games] [seed]
"""
import random
import sys
from collections import Counter

MAX_DAY = 8


def play(rng: random.Random):
    mafia, doctor = 0, 1
    alive = set(range(7))
    doc_self_last = False
    saves = 0
    day = 0
    while True:
        day += 1
        # night
        target = rng.choice([p for p in alive if p != mafia])
        protect = None
        if doctor in alive:
            protect = rng.choice([p for p in alive if not (p == doctor and doc_self_last)])
            doc_self_last = protect == doctor
        if target == protect:
            saves += 1
        else:
            alive.discard(target)
        if len(alive) - 1 <= 1:
            return "mafia", saves, day
        if day >= MAX_DAY:
            return "town", saves, day
        # day vote (random)
        votes = Counter(rng.choice([p for p in alive if p != v]) for v in alive)
        top = max(votes.values())
        leaders = [p for p, c in votes.items() if c == top]
        if len(leaders) == 1:
            alive.discard(leaders[0])
            if leaders[0] == mafia:
                return "town", saves, day
        if len(alive) - 1 <= 1:
            return "mafia", saves, day


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rng = random.Random(seed)
    results = [play(rng) for _ in range(n)]
    town = sum(r[0] == "town" for r in results) / n
    saves = sum(r[1] for r in results) / n
    days = sorted(r[2] for r in results)
    print(f"games={n} town_win={town:.3f} mafia_win={1 - town:.3f} "
          f"saves_per_game={saves:.2f} median_day={days[n // 2]} "
          f"p_day8={sum(d >= MAX_DAY for d in days) / n:.2e}")


if __name__ == "__main__":
    main()
