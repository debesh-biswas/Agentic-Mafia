"""Live HTML dashboard for a season: one self-contained file, rebuilt from the run folder (no LLM, no writes elsewhere)."""
import csv
import json
import os
import sqlite3
from datetime import datetime
from html import escape as esc
from pathlib import Path

import yaml

from mafia.logs.events import EventLog
from mafia.renderers.text import TextRenderer, visible

BLOCK = 5
REASON = {"parity": "parity", "mafia_eliminated": "mafia voted out", "day_limit": "day limit"}
TRANSCRIPT = {"night_action", "night_resolution", "morning_announcement", "speech", "pass", "last_words", "role_reveal"}
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mute:#667085;--line:#e3e6ec;--town:#1f7a4d;--mafia:#b42318;--warn:#b54708}
@media(prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2030;--fg:#e8ebf2;--mute:#98a2b3;--line:#2c3347;--town:#4cc38a;--mafia:#ff7b72;--warn:#f5a623}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,sans-serif}
main{max-width:980px;margin:0 auto}h1{margin:0 0 4px;font-size:1.5rem}h2{margin:28px 0 10px;font-size:1.15rem}
.mute{color:var(--mute);font-size:.9rem}.stats{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.stat b{display:block;font-size:1.1rem}.stat span{color:var(--mute);font-size:.8rem}
.wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--mute);font-weight:600;font-size:.85rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:10px}
.card h3{margin:0 0 4px;font-size:1rem}.town{color:var(--town)}.mafia{color:var(--mafia)}.warn{color:var(--warn)}
details{margin-top:8px}summary{cursor:pointer;color:var(--mute)}pre{white-space:pre-wrap;font:.85rem/1.5 ui-monospace,monospace;margin:8px 0 0}
.who{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}ul{margin:4px 0;padding-left:20px}
"""


def _csv(path):
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _names(run_dir):
    p = yaml.safe_load((run_dir / "config" / "personas.yaml").read_text())["personas"]
    return {x["agent_id"]: x["name"] for x in p}


def _cut(s, n=140):
    return s if len(s) <= n else s[:n - 1] + "…"


def _blocks(games):
    rows = ""
    for i in range(0, len(games), BLOCK):
        b = games[i:i + BLOCK]
        town = sum(g["winner"] == "town" for g in b)
        days = sum(int(g["days"]) for g in b) / len(b)
        bad = sum(int(g["format_errors"]) + int(g["fallbacks"]) for g in b)
        label = f"{b[0]['game']}–{b[-1]['game']}" if len(b) > 1 else b[0]["game"]
        rows += (f"<tr><td>{label}{'' if len(b) == BLOCK else ' (so far)'}</td><td>{town}</td><td>{len(b) - town}</td>"
                 f"<td>{days:.1f}</td><td>${sum(float(g['cost_usd']) for g in b):.4f}</td>"
                 f"<td class='{'warn' if bad else ''}'>{bad}</td></tr>")
    return ("<h2>Every 5 games</h2><div class='wrap'><table><tr><th>Games</th><th>Town won</th><th>Mafia won</th>"
            f"<th>Avg length (days)</th><th>Cost</th><th>Problems</th></tr>{rows}</table></div>")


def _scoreboard(agents, names):
    tot, rec = {}, {}
    for r in agents:
        a = r["agent"]
        tot[a] = tot.get(a, 0) + int(r["points_total"])
        w = rec.setdefault(a, {}).setdefault(r["role"], [0, 0])
        w[0 if r["team_won"] == "True" else 1] += 1
    rows = "".join(
        f"<tr><td>{esc(names.get(a, a))}</td><td>{tot[a]}</td><td>{esc(', '.join(f'{k} {w}–{l}' for k, (w, l) in sorted(rec[a].items())))}</td></tr>"
        for a in sorted(tot, key=lambda a: (-tot[a], names.get(a, a))))
    return f"<h2>Scoreboard</h2><div class='wrap'><table><tr><th>Player</th><th>Points</th><th>Record (won–lost) by role</th></tr>{rows}</table></div>"


def _game(g, names, run_dir):
    n = lambda a: esc(names.get(a, a))
    won = g["winner"] == "town"
    head = (f"Game {g['game']} — <span class='{'town' if won else 'mafia'}'>{'Town' if won else 'Mafia'} won on day {g['days']}</span>"
            f" ({esc(REASON.get(g['end_reason'], g['end_reason']))})")
    bad = int(g["format_errors"]) + int(g["fallbacks"])
    steps, lines, votes = [], [], []
    try:
        events = visible(EventLog(run_dir, int(g["game"]) - 1).read(), "truth")
        tr = TextRenderer(names)
        for e in events:
            if e.type == "night_resolution":
                steps.append(f"Night {e.day}: " + (f"{n(e.payload['eliminated'])} eliminated" if e.payload["eliminated"] else "saved by doctor"))
            elif e.type == "vote":
                votes.append(f"{names.get(e.actor, e.actor)}→{names.get(e.targets[0], e.targets[0])}")
            elif e.type == "vote_result":
                out = "tie, nobody voted out" if e.payload["tie"] else f"{names.get(e.payload['eliminated'])} voted out"
                steps.append(f"Day {e.day}: " + esc(out))
                lines.append(f"Votes: {', '.join(votes)}. {out}.")
                votes = []
            elif e.type in TRANSCRIPT and (r := tr.line(e)):
                lines.append(r[0])
    except Exception:  # missing or torn log: show the summary only
        lines = ["Transcript not available yet."]
    return (f"<div class='card'><h3>{head}</h3><div>Mafia: <b>{n(g['mafia'])}</b> · Doctor: <b>{n(g['doctor'])}</b></div>"
            f"<div class='mute'>{' · '.join(steps)}</div>"
            + (f"<div class='warn'>{bad} problem(s) with model answers</div>" if bad else "")
            + f"<details><summary>Full transcript</summary><pre>{esc(chr(10).join(lines))}</pre></details></div>")


def _beliefs(run_dir, names):
    out = ""
    for a, name in names.items():
        path = run_dir / "memory" / f"{a}.sqlite"
        try:  # read-only short-lived connection; the season may be writing
            db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
            try:
                c = dict(db.execute("SELECT type, COUNT(*) FROM memories WHERE status='active' GROUP BY type"))
                me = db.execute("SELECT text FROM summaries WHERE subject='self' ORDER BY game DESC LIMIT 1").fetchone()
                top = db.execute("SELECT text FROM memories WHERE status='active' AND type IN ('belief','lesson') "
                                 "ORDER BY importance DESC, id DESC LIMIT 3").fetchall()
            finally:
                db.close()
            body = (f"<div class='mute'>{c.get('belief', 0)} beliefs · {c.get('lesson', 0)} lessons · "
                    f"{c.get('reflection', 0)} reflections · {c.get('episodic', 0)} game summaries</div>"
                    f"<div>{esc(_cut(me[0])) if me else 'No self summary yet.'}</div>"
                    f"<ul>{''.join(f'<li>{esc(_cut(t[0]))}</li>' for t in top)}</ul>")
        except sqlite3.Error:
            body = "<div class='mute'>updating…</div>"
        out += f"<div class='card'><h3>{esc(name)}</h3>{body}</div>"
    return f"<h2>What the players believe now</h2><div class='who'>{out}</div>"


def write_dashboard(run_dir) -> Path:
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    names = _names(run_dir)
    games, agents = _csv(run_dir / "games.csv"), _csv(run_dir / "agents.csv")
    cfg, group = man["run_config"], man["group"]
    pin = ", ".join(man.get("provider_pin") or [])
    done, secs = len(games), sum(float(g["seconds"]) for g in games)
    stats = [(man["run_id"], "run"), (group, "group"), (f"Game {done} of {cfg['games']}" if done else "Waiting for the first game…", "progress"),
             (f"{man['model']}{' via ' + pin if pin else ''}", "model"), (datetime.now().strftime("%H:%M:%S"), "last updated"),
             (f"${sum(float(g['cost_usd']) for g in games):.4f}", "total cost"), (str(sum(int(g["llm_calls"]) for g in games)), "model calls"),
             (f"{secs / done / 60:.1f}" if done else "–", "avg minutes per game")]
    body = f"<h1>Mafia season</h1><div class='stats'>{''.join(f'<div class=stat><b>{esc(v)}</b><span>{k}</span></div>' for v, k in stats)}</div>"
    if done:
        cards = [_game(g, names, run_dir) for g in reversed(games)]
        old = f"<details><summary>Older games ({len(cards) - 5})</summary>{''.join(cards[5:])}</details>" if len(cards) > 5 else ""
        body += _blocks(games) + _scoreboard(agents, names) + f"<h2>Games, newest first</h2>{''.join(cards[:5])}{old}"
    if group == "memory":
        body += _beliefs(run_dir, names)
    if sum(int(g["fallbacks"]) for g in games):
        body += "<p class='warn'>Some answers were invalid and replaced by a random legal move.</p>"
    html = ("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<meta http-equiv=refresh content=60><title>Mafia season {esc(man['run_id'])}</title><style>{CSS}</style><main>{body}</main></html>")
    out = run_dir / "dashboard.html"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(html)
    os.replace(tmp, out)
    return out
