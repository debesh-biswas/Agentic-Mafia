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
from mafia.renderers.text import visible

BLOCK = 5
REASON = {"parity": "parity", "mafia_eliminated": "mafia voted out", "day_limit": "day limit"}
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
.chat{margin:4px 0}.chat b{margin-right:6px}.tag{color:var(--mute);font-size:.8rem;margin-left:6px}.priv{color:var(--mute);font-style:italic}
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
    steps = []
    try:
        for e in visible(EventLog(run_dir, int(g["game"]) - 1).read(), "truth"):
            if e.type == "night_resolution":
                steps.append(f"Night {e.day}: " + (f"{n(e.payload['eliminated'])} eliminated" if e.payload["eliminated"] else "saved by doctor"))
            elif e.type == "vote_result":
                steps.append(f"Day {e.day}: " + ("tie, nobody voted out" if e.payload["tie"] else f"{n(e.payload['eliminated'])} voted out"))
    except Exception:  # missing or torn log: show the summary only
        pass
    return (f"<div class='card'><h3>{head}</h3><div>Mafia: <b>{n(g['mafia'])}</b> · Doctor: <b>{n(g['doctor'])}</b></div>"
            f"<div class='mute'>{' · '.join(steps)}</div>"
            + (f"<div class='warn'>{bad} problem(s) with model answers</div>" if bad else "")
            + f"<div><a href='games/{int(g['game']):04d}.html'>Open full game →</a></div></div>")


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


def _fate(events, names):
    fate = {}
    for e in events:
        if e.type == "night_resolution" and e.payload["eliminated"]:
            fate[e.payload["eliminated"]] = f"eliminated night {e.day}"
        elif e.type == "vote_result" and e.payload["eliminated"]:
            fate[e.payload["eliminated"]] = f"voted out day {e.day}"
    return fate


FALLBACK_WHAT = {"speak": "speech", "vote": "vote", "night": "night choice", "last_words": "last words", "plan": "plan"}


def _problem(e, n):
    p = e.payload
    what = FALLBACK_WHAT.get(p.get("call_type"), str(p.get("call_type")).replace("_", " "))
    r = str(p.get("reason"))
    why = "cut off (too long)" if r == "truncated" else f"not usable ({r})"
    tail = " — a random legal choice was used" if p.get("chosen") not in (None, "None") else ""
    return f"<li>{n(e.actor)}'s {esc(what)} was {esc(why)}{esc(tail)}</li>"


def _thoughts(pid, evs, n):
    by = lambda t: [e for e in evs if e.actor == pid and e.type == t]
    ref, summ = by("reflection"), by("summary_update")
    r = ref[0].payload if ref else {}
    h = "".join(f"<li>{esc(i['text'])} <span class='tag'>{esc(i['kind'])} · {esc(i['action'])} · {esc(i['confidence'])}</span></li>"
                for i in r.get("insights", []))
    body = ("<b>Questions</b><ul>" + "".join(f"<li>{esc(q)}</li>" for q in r.get("questions", [])) + "</ul>"
            f"<b>Insights</b><ul>{h}</ul><b>Did my beliefs help or hurt?</b><p>{esc(r.get('helped_or_hurt', ''))}</p>"
            f"<b>Game summary</b><p>{esc(r.get('game_summary', ''))}</p>")
    if summ:
        sp = summ[0].payload
        body += ("<b>Impressions now</b><ul>" + "".join(f"<li>{n(k)}: {esc(v)}</li>" for k, v in sp["friends"].items())
                 + f"<li>Self: {esc(sp['self'])}</li></ul>")
    extra = [x for x in by("memory_write")][len(r.get("insights", [])):]
    more = ([f"<li>Memory {esc(x.payload['action'])} ({esc(x.payload['type'])}): {esc(x.payload['text'])}</li>" for x in extra]
            + [f"<li>Replaced memory {x.payload['old_id']} with {x.payload['new_id']}</li>" for x in by("memory_supersede")]
            + [f"<li>Maintenance: merged {esc(x.payload['merged'])}, faded {esc(x.payload['faded'])}</li>" for x in by("memory_maintenance")])
    if more:
        body += f"<b>Deep reflection, compression, maintenance</b><ul>{''.join(more)}</ul>"
    return (f"<details><summary><b>{n(pid)}</b> — {esc(_cut(r.get('helped_or_hurt', '')))}</summary>{body}</details>")


def write_game_page(run_dir, game_no) -> Path:
    run_dir = Path(run_dir)
    names = _names(run_dir)
    n = lambda a: esc(names.get(a, a)) if a else ""
    events = EventLog(run_dir, game_no - 1).read()
    pub = [e for e in events if e.visibility != "debug"]
    first = lambda t: next((e for e in pub if e.type == t), None)
    end, reveal, score, start = first("game_end"), first("role_reveal"), first("score"), first("game_start")
    roles = reveal.payload["roles"] if reveal else {}
    seats = list(start.payload["seats"]) if start else list(names)
    fate = _fate(pub, names)
    calls = [e for e in events if e.type == "llm_call"]
    cost = sum(e.payload.get("cost") or 0 for e in calls)
    if end:
        won = end.payload["winner"] == "town"
        head = (f"Game {game_no} — <span class='{'town' if won else 'mafia'}'>{'Town' if won else 'Mafia'} won on day "
                f"{end.payload['days']}</span> ({esc(REASON.get(end.payload['end_reason'], end.payload['end_reason']))})")
    else:
        head = f"Game {game_no} — unfinished"
    pts = score.payload["points"] if score else {}
    rows = "".join(f"<tr><td>{n(a)}</td><td>{esc(roles.get(a, '?'))}</td><td>{esc(fate.get(a, 'survived'))}</td>"
                   f"<td>{pts.get(a, {}).get('points_total', '–')}</td></tr>" for a in seats)
    out = [f"<p><a href='../dashboard.html'>← Back to dashboard</a></p><h1>{head}</h1>",
           f"<div class='mute'>{len(calls)} model calls · ${cost:.4f}</div>",
           "<div class='wrap'><table><tr><th>Player</th><th>Role</th><th>Fate</th><th>Points this game</th></tr>" + rows + "</table></div>"]
    plans = [e for e in pub if e.type == "plan"]
    if plans:
        out.append("<h2>Before the game</h2><p class='mute'>Private plans</p><ul>" + "".join(
            f"<li><b>{n(e.actor)}</b> <span class='priv'>(private)</span>: {esc(e.payload['plan'])}</li>" for e in plans) + "</ul>")
    sect = None
    for e in pub:
        t, p = e.type, e.payload
        key = {"night_notes": f"Night {e.day}", "night_action": f"Night {e.day}", "night_resolution": f"Night {e.day}",
               "morning_announcement": f"Morning, day {e.day}", "speech": f"Day {e.day} — discussion", "pass": f"Day {e.day} — discussion",
               "vote": f"Day {e.day} — votes", "vote_result": f"Day {e.day} — votes", "last_words": f"Day {e.day} — votes"}.get(t)
        if key is None:
            continue
        if key != sect:
            out.append(f"<h2>{key}</h2>")
            sect = key
        if t == "night_notes":
            out.append(f"<div class='chat'><b>{n(e.actor)}</b> <span class='priv'>{esc(p['notes'])}</span><span class='tag'>wants to speak: {p['urgency']}</span></div>")
        elif t == "night_action":
            verb = "targets" if p["action"] == "target" else "protects"
            out.append(f"<div class='priv'>{n(e.actor)} ({'mafia' if p['action'] == 'target' else 'doctor'}) {verb} {n(e.targets[0])}</div>")
        elif t == "night_resolution":
            res = (f"{n(p['eliminated'])} was eliminated" if p["eliminated"] else f"{n(p['target'])} was saved by the doctor")
            out.append(f"<div><b>Result:</b> {res}</div>")
        elif t == "morning_announcement":
            out.append(f"<div>{esc(p['text'])}</div>")
        elif t == "speech":
            tags = "".join([f"<span class='tag'>accuses {', '.join(n(a) for a in p['accuses'])}</span>" if p["accuses"] else "",
                            "<span class='tag'>replying to an accusation</span>" if p["priority_reply"] else "",
                            f"<span class='tag'>{esc(p['expression'])}</span>" if p.get("expression") else ""])
            out.append(f"<div class='chat'><b>{n(e.actor)}</b>{esc(p['message'])}{tags}</div>")
        elif t == "pass":
            out.append(f"<div class='mute'>{n(e.actor)} passes</div>")
        elif t == "vote":
            out.append(f"<div>{n(e.actor)} → {n(e.targets[0])}</div>")
        elif t == "vote_result":
            out.append("<div><b>Result:</b> " + ("tie, nobody voted out" if p["tie"] else f"{n(p['eliminated'])} was voted out") + "</div>")
        elif t == "last_words":
            out.append(f"<div class='chat'><b>{n(e.actor)}</b>(last words) {esc(p['message'])}</div>")
    if end:
        out.append(f"<h2>Game over</h2><div>{'Town' if end.payload['winner'] == 'town' else 'Mafia'} wins. Roles: "
                   + esc(", ".join(f"{names.get(a, a)} {r}" for a, r in roles.items())) + "</div>")
    refl = [e for e in pub if e.phase == "reflect" and e.type != "leaderboard"]
    if refl:
        out.append("<h2>After the game — what each player thought</h2>" + "".join(_thoughts(a, refl, n) for a in seats))
    bad = [e for e in events if e.type == "fallback"]
    if bad:
        out.append("<h2>Problems</h2><ul>" + "".join(_problem(e, n) for e in bad) + "</ul>")
    html = ("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Game {game_no}</title><style>{CSS}</style><main>{''.join(out)}</main></html>")
    path = run_dir / "games" / f"{game_no:04d}.html"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(html)
    os.replace(tmp, path)
    return path


def _pages(run_dir, games):
    for g in games:
        k = int(g["game"])
        src, page = run_dir / "games" / f"{k:04d}.jsonl", run_dir / "games" / f"{k:04d}.html"
        try:
            if src.exists() and (not page.exists() or src.stat().st_mtime > page.stat().st_mtime):
                write_game_page(run_dir, k)
        except Exception as e:  # one bad log must not block the dashboard
            print(f"warning: game page {k} not written ({type(e).__name__}: {e})")


def write_dashboard(run_dir) -> Path:
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    names = _names(run_dir)
    games, agents = _csv(run_dir / "games.csv"), _csv(run_dir / "agents.csv")
    _pages(run_dir, games)
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
