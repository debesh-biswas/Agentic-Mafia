"""DRD 14: markdown report from runs/<id>/games.csv + agents.csv (+ memory DBs). Stdlib + numpy only."""
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

WINDOW = 50  # reporting choice, not a game rule
ROLES = ("mafia", "doctor", "villager")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def f(x):
    return "-" if x != x else f"{x:.3f}"


def load(d: Path) -> dict:
    rd = lambda n: list(csv.DictReader((d / n).open(newline="")))
    G, A = rd("games.csv"), rd("agents.csv")
    for r in G:
        r["game"] = int(r["game"])
        for k in ("saves", "format_errors", "fallbacks", "self_leaks"):
            r[k] = int(r[k])
    for r in A:
        r["game"] = int(r["game"])
        for k in ("points_total", "votes_cast", "votes_on_mafia", "days_alive"):
            r[k] = int(r[k])
        r["won"] = r["team_won"] == "True"
    return dict(id=d.name, dir=d, group=G[0]["group"], G=G, A=A)


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def by_window(rows):
    w = defaultdict(list)
    for r in rows:
        w[(r["game"] - 1) // WINDOW].append(r)
    return dict(sorted(w.items()))


def label(i):
    return f"{i * WINDOW + 1}-{(i + 1) * WINDOW}"


def windowed(runs, fn, cols):
    """One table: a row per window, one column per (run, col); fn(G_rows, A_rows) -> list of values per col."""
    ws = sorted({i for r in runs for i in by_window(r["G"])})
    body = []
    for i in ws:
        cells = []
        for r in runs:
            g, a = by_window(r["G"]).get(i, []), by_window(r["A"]).get(i, [])
            cells += [f(v) for v in fn(g, a)] if g else ["-"] * len(cols)
        body.append([label(i)] + cells)
    allc = []
    for r in runs:
        allc += [f(v) for v in fn(r["G"], r["A"])]
    head = ["games"] + [f"{r['id']} {c}" for r in runs for c in cols]
    return table(head, [["all"] + allc] + body)


def baseline(runs):
    """Mean points_total per role over all control games."""
    pts = defaultdict(list)
    for r in runs:
        if r["group"] == "control":
            for a in r["A"]:
                pts[a["role"]].append(a["points_total"])
    return {k: mean(v) for k, v in pts.items()}


def adjust(runs, base):
    for r in runs:
        for a in r["A"]:
            a["adj"] = a["points_total"] - base.get(a["role"], float("nan"))


def paired(mem, ctl, n=2000):
    pg = lambda r: {g: mean(a["adj"] for a in r["A"] if a["game"] == g) for g in {a["game"] for a in r["A"]}}
    m, c = pg(mem), pg(ctl)
    d = np.array([m[g] - c[g] for g in sorted(m.keys() & c.keys())])
    if not len(d):
        return None
    rng = np.random.default_rng(0)
    boots = rng.choice(d, (n, len(d))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return len(d), d.mean(), lo, hi


def memory_stats(r, names):
    out = []
    for p in sorted((r["dir"] / "memory").glob("*.sqlite")):
        db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        q = lambda s: db.execute(s).fetchall()
        n = lambda t, s="active": q(f"SELECT COUNT(*) FROM memories WHERE type='{t}' AND status='{s}'")[0][0]
        top = "; ".join(f"{t[:80]} ({k}x)" for t, k in q(
            "SELECT text, times_retrieved FROM memories WHERE times_retrieved>0 ORDER BY times_retrieved DESC, id LIMIT 3"))
        out.append([names.get(p.stem, p.stem), n("belief"), n("lesson"),
                    q("SELECT COUNT(*) FROM memories WHERE status='superseded'")[0][0], top or "-"])
        db.close()
    return out


def build(runs: list[dict], names: dict[str, str]) -> str:
    base = baseline(runs)
    ctl = [r for r in runs if r["group"] == "control"]
    mem = [r for r in runs if r["group"] == "memory"]
    adjust(runs, base)
    o = ["# Mafia run report\n", "Runs: " + ", ".join(f"{r['id']} ({r['group']}, {len(r['G'])} games)" for r in runs) + "\n",
         f"Windows are non-overlapping blocks of {WINDOW} games; every \"rolling mean\" below is the mean over each block.\n"]

    o.append("## 1. Role win rates\n")
    cols = [f"{x} win" for x in ROLES]
    o.append(windowed(runs, lambda g, a: [mean(x["won"] for x in a if x["role"] == ro) for ro in ROLES], cols))

    o.append("## 2. Role-adjusted points\n")
    if ctl:
        o.append("points_total minus the control-group mean for the same role ("
                 + ", ".join(f"{k} {v:.3f}" for k, v in sorted(base.items())) + ").\n")
        o.append(windowed(runs, lambda g, a: [mean(x["adj"] for x in a)], ["adj points"]))
    else:
        o.append("Skipped: no control run given, so there is no role baseline.\n")

    o.append("## 3. Vote accuracy, save rate, mafia days survived\n")

    def m3(g, a):
        town = [x for x in a if x["role"] != "mafia"]
        return [sum(x["votes_on_mafia"] for x in town) / max(1, sum(x["votes_cast"] for x in town)),
                mean(x["saves"] for x in g), mean(x["days_alive"] for x in a if x["role"] == "mafia")]
    o.append(windowed(runs, m3, ["vote acc", "save rate", "mafia days"]))

    o.append("## 4. Per-agent table\n")
    for r in runs:
        n, rows = len(r["G"]), []
        split = 100 if n >= 200 else n // 2
        early = lambda x: x["game"] <= split
        late = (lambda x: x["game"] > n - 100) if n >= 200 else (lambda x: x["game"] > split)
        o.append(f"### {r['id']}\n\nEarly = games 1-{split}, late = " +
                 (f"last 100 ({n - 99}-{n})" if n >= 200 else f"second half ({split + 1}-{n}); fewer than 200 games, so halves") + ".\n")
        for ag in sorted({x["agent"] for x in r["A"]}):
            A = [x for x in r["A"] if x["agent"] == ag]
            cnt = "/".join(str(sum(x["role"] == ro for x in A)) for ro in ROLES)
            win = "/".join(f(mean(x["won"] for x in A if x["role"] == ro)) for ro in ROLES)
            adj = lambda s: f(mean(x["adj"] for x in A if s(x))) if ctl else "-"
            rows.append([names.get(ag, ag), cnt, win, adj(lambda x: True), adj(early), adj(late)])
        o.append(table(["agent", "games m/d/v", "win rate m/d/v", "adj points", "adj early", "adj late"], rows))

    o.append("## 5. Reliability\n")
    o.append(windowed(runs, lambda g, a: [mean(x["format_errors"] for x in g), mean(x["fallbacks"] for x in g),
                                           mean(x["self_leaks"] for x in g)], ["fmt err/game", "fallbacks/game", "self-leaks/game"]))

    o.append("## 6. Memory stats\n")
    shown = False
    for r in mem:
        rows = memory_stats(r, names)
        if rows:
            shown = True
            o.append(f"### {r['id']} (end of run)\n")
            o.append(table(["agent", "active beliefs", "active lessons", "superseded", "top 3 retrieved"], rows))
    if not shown:
        o.append("No memory runs or memory databases found.\n")

    o.append("## 7. Paired comparison (memory - control)\n")
    p = paired(mem[0], ctl[0]) if mem and ctl else None
    if p:
        o.append(f"{mem[0]['id']} vs {ctl[0]['id']}: {p[0]} games paired by game number. Per-game difference of mean "
                 f"role-adjusted points across the 7 agents.\n\nmean {p[1]:.3f}, bootstrap 95% CI [{p[2]:.3f}, {p[3]:.3f}] "
                 "(2,000 resamples, seed 0)\n")
    else:
        o.append("Skipped: needs one memory run and one control run.\n")
    return "\n".join(o)
