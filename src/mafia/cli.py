import typer

app = typer.Typer(help="LLM agents playing Mafia.", no_args_is_help=True)

O = typer.Option(...)  # required option


def _todo():
    typer.echo("not implemented yet")
    raise typer.Exit(1)


@app.command()
def sim(games: int = 200000, seed: int = 1, save_run: str | None = None):
    """Random-bot balance check."""
    import statistics
    import time

    from mafia.agents.random_agent import random_agents
    from mafia.config import ROOT, Personas, Rules, load
    from mafia.engine.game import Game
    from mafia.logs.events import EventLog

    rules = load(Rules, ROOT / "config/rules.yaml")
    personas = load(Personas, ROOT / "config/personas.yaml").personas
    ids = [p.agent_id for p in sorted(personas, key=lambda p: p.seat)]
    t0, town, saves, days = time.time(), 0, 0, []
    for i in range(games):
        g = Game(rules, personas, random_agents(rules, seed, i, ids), save_run or "sim", i, seed, "sim", 0, 0,
                 record=bool(save_run), concurrent=False)
        r = g.play()
        if save_run:
            EventLog(ROOT / "runs" / save_run, i).write(g.events)
        town += r.winner == "town"
        saves += r.saves
        days.append(r.days)
    typer.echo(f"games={games} town_win={town / games:.3f} mafia_win={1 - town / games:.3f} "
               f"saves_per_game={saves / games:.2f} median_day={statistics.median(days):g} "
               f"({time.time() - t0:.0f}s)")


def _run(run_config, resume, mock, games=None, on_game=None):
    from mafia.runner.season import RunExists, run_season
    try:
        run_season(run_config, mock=mock, resume=resume, games=games, on_game=on_game)
    except RunExists as e:
        typer.echo(str(e))
        raise typer.Exit(1)


@app.command()
def play(run_config: str = O, games: int = 1, view: str = "public", resume: bool = False,
         mock: bool = typer.Option(False, hidden=True)):
    """Play games with a run config, printing each transcript."""
    from mafia.config import Personas, load, load_run
    from mafia.renderers.text import TextRenderer

    run = load_run(run_config)
    names = {p.agent_id: p.name for p in load(Personas, run.personas_config).personas}
    if view not in {"public", "truth", *names}:
        typer.echo(f"unknown view {view!r}; use public, truth or one of: {', '.join(names)}")
        raise typer.Exit(1)
    _run(run_config, resume, mock, games, lambda events, g: TextRenderer(names).render_game(events, view))


@app.command()
def season(run_config: str = O, resume: bool = False, mock: bool = typer.Option(False, hidden=True)):
    """Run a full season."""
    _run(run_config, resume, mock)


@app.command()
def calibrate(run: str = O):
    """Win rates and proposed scoring_v1.yaml."""
    _todo()


@app.command()
def score(run: str = O, scoring: str = O):
    """Recompute points for a run from its logs (logs are not modified)."""
    import json

    from mafia.config import ROOT, Personas, Scoring, load
    from mafia.logs.events import Event
    from mafia.scoring import leaderboard
    from mafia.scoring.score import KEYS, score as rescore

    cfg = load(Scoring, scoring)
    manifest = ROOT / "runs" / run / "manifest.json"
    if not manifest.exists():
        typer.echo(f"no manifest for run {run!r}")
        raise typer.Exit(1)
    pfile = json.loads(manifest.read_text())["run_config"]["personas_config"]
    names = {p.agent_id: p.name for p in load(Personas, pfile).personas}
    files = sorted((ROOT / "runs" / run / "games").glob("*.jsonl"))
    if not files:
        typer.echo(f"no game logs for run {run!r}")
        raise typer.Exit(1)
    per_game = [rescore([Event.model_validate_json(l) for l in f.read_text().splitlines() if l], cfg) for f in files]
    tot = leaderboard.standings(per_game)
    typer.echo(f"{run}: {len(files)} games, scoring v{cfg.version}")
    for a in sorted(tot, key=lambda a: (-tot[a], names[a])):
        parts = " ".join(f"{k.removeprefix('points_')}={sum(g[a][k] for g in per_game)}" for k in KEYS)
        typer.echo(f"{names[a]:8} total={tot[a]:5}  {parts}")


@app.command()
def replay(run: str = O, game: int = O, view: str = "public"):
    """Replay a recorded game."""
    from mafia.config import ROOT, Personas, load
    from mafia.logs.events import EventLog
    from mafia.renderers.text import TextRenderer

    names = {p.agent_id: p.name for p in load(Personas, ROOT / "config/personas.yaml").personas}
    if view not in {"public", "truth", *names}:
        typer.echo(f"unknown view {view!r}; use public, truth or one of: {', '.join(names)}")
        raise typer.Exit(1)
    log = EventLog(ROOT / "runs" / run, game - 1)
    if not log.path.exists():
        typer.echo(f"no such game log: {log.path}")
        raise typer.Exit(1)
    TextRenderer(names).render_game(log.read(), view)


@app.command()
def inspect(
    run: str = O,
    agent: str = O,
    beliefs: bool = False,
    lessons: bool = False,
    episodic: bool = False,
    summaries: bool = False,
    superseded: bool = False,
    at_game: int | None = None,
    runs_dir: str | None = typer.Option(None, hidden=True),
):
    """Inspect an agent's memory."""
    import json
    from pathlib import Path
    from mafia.config import ROOT, Personas, load
    from mafia.memory.embed import FakeEmbedder
    from mafia.memory.store import MemoryStore

    d = (Path(runs_dir) if runs_dir else ROOT / "runs") / run
    if at_game is None:
        path = d / "memory" / f"{agent}.sqlite"
    else:
        snaps = sorted((int(p.name[1:]), p) for p in (d / "memory_snapshots").glob("g[0-9]*")
                       if int(p.name[1:]) <= at_game and (p / f"{agent}.sqlite").exists())
        if not snaps:
            typer.echo(f"no memory snapshot at or before game {at_game} for run {run!r}; snapshots are written every "
                       "snapshot_memory_every games")
            raise typer.Exit(1)
        path = snaps[-1][1] / f"{agent}.sqlite"
    if not path.exists():
        typer.echo(f"no memory store for agent {agent!r} in run {run!r} (control group runs have none)")
        raise typer.Exit(1)
    pfile = json.loads((d / "manifest.json").read_text())["run_config"]["personas_config"]
    names = {p.agent_id: p.name for p in load(Personas, pfile).personas}
    st = MemoryStore(path, FakeEmbedder())  # embedder is never used for reading
    try:
        if summaries:
            for k, t in sorted(st.get_summaries().items()):
                typer.echo(f"{names.get(k, k)}: {t}")
            return
        if superseded:
            rows = st.active(status="superseded")
        else:
            types = [t for t, on in (("belief", beliefs), ("lesson", lessons), ("episodic", episodic)) if on]
            rows = st.active(types or None)
        for t in dict.fromkeys(m.type for m in rows):
            typer.echo(f"== {t} ==")
            for m in (m for m in rows if m.type == t):
                about = ",".join(names.get(a, a) for a in m.about)
                extra = f" about={about}" if about else ""
                extra += f" role={m.role_context}" if m.role_context else ""
                extra += f" conf={m.confidence}" if m.confidence else ""
                if m.type == "lesson":
                    extra += f" won {m.wins_when_applied}/{m.games_applied}"
                if m.superseded_by:
                    extra += f" -> m{m.superseded_by}"
                typer.echo(f"[m{m.id}] g{m.game_created}-{m.game_updated} imp={m.importance}{extra}: {m.text}")
                for g, old, _ in st.versions(m.id):
                    typer.echo(f"    (before game {g}: {old})")
    finally:
        st.close()


@app.command()
def report(runs: list[str] = O, runs_dir: str | None = typer.Option(None, hidden=True)):
    """Markdown report for runs."""
    from datetime import datetime
    from pathlib import Path

    from mafia.config import ROOT, Personas, load
    from mafia.metrics import report as rep

    base = Path(runs_dir) if runs_dir else ROOT / "runs"
    for r in runs:
        if not (base / r / "games.csv").exists():
            typer.echo(f"no games.csv for run {r!r}")
            raise typer.Exit(1)
    names = {p.agent_id: p.name for p in load(Personas, ROOT / "config/personas.yaml").personas}
    out = base / f"report_{datetime.now():%Y%m%d-%H%M%S}.md"
    out.write_text(rep.build([rep.load(base / r) for r in runs], names))
    typer.echo(str(out))


@app.command()
def scene(run: str = O, game: int = O, out: str = O, runs_dir: str | None = typer.Option(None, hidden=True)):
    """Export a scene script (stub)."""
    import json
    from pathlib import Path

    from mafia.config import ROOT, Personas, load
    from mafia.logs.events import EventLog
    from mafia.renderers.scene_script import beats

    log = EventLog((Path(runs_dir) if runs_dir else ROOT / "runs") / run, game - 1)
    if not log.path.exists():
        typer.echo(f"no such game log: {log.path}")
        raise typer.Exit(1)
    seats = {p.agent_id: p.seat for p in load(Personas, ROOT / "config/personas.yaml").personas}
    Path(out).write_text(json.dumps(beats(log.read(), seats), indent=2))
    typer.echo(out)


_CHECKS = [  # (schema, call_type, tiny prompt)
    ("PlanDecision", "plan", "It is the start of day 1. Write your short plan for the game."),
    ("NightDecision", "night", "Night 1. You are a villager. Living players: Ann, Bo, Cy. Write notes, an urgency 0-4; no target."),
    ("SpeakDecision", "speak", "Day 1 discussion. Living: Ann, Bo, Cy. Say something short, or null, with urgency 0-4."),
    ("VoteDecision", "vote", "Vote now. Living: Bo, Cy. Pick one name to eliminate."),
    ("LastWordsDecision", "last_words", "You were eliminated. Say your last words in under 20 words."),
    ("Questions", "reflect", "The game is over. Ask 1-3 questions to reflect on."),
    ("ReflectInsights", "reflect", "The game is over. Give 1 insight (kind lesson), keep_ids [], helped_or_hurt, game_summary and summary_importance."),
    ("Summaries", "summaries", "Summarize what you think of your friends Bo and Cy (one sentence each) and of yourself."),
    ("DeepReflect", "deep_reflect", "Give 1 insight (kind lesson) about your play across several games."),
    ("Compressed", "compress", "Merge these memories into one summary with an importance 1-10: 'Bo lied once.' 'Bo lied twice.'"),
]


@app.command(hidden=True)
def llm_check(model_config: str = O, model: str | None = None):
    """Send one tiny call per decision schema to the configured endpoint; exit 1 if any fail."""
    import json
    import time

    from mafia.config import ModelCfg, load
    from mafia.llm import schemas
    from mafia.llm.client import LLMClient

    cfg = load(ModelCfg, model_config)
    if model:
        cfg.model = model
    client, failed = LLMClient(cfg), 0
    for name, call_type, prompt in _CHECKS:
        schema = getattr(schemas, name)
        t0 = time.monotonic()
        try:
            _, rec = client.call(call_type, "You are Ann, playing a game of Mafia. Reply with JSON only.",
                                 f"{prompt}\nReturn JSON matching: {json.dumps(schema.model_json_schema())}",
                                 schema, "ann")
            status, retries = "PASS", rec["format_errors"]
        except Exception as e:  # any failure is a FAIL line, not a crash
            status, retries = f"FAIL ({type(e).__name__}: {str(e)[:120]})", getattr(e, "record", {}).get("format_errors", "-")
            failed += 1
        typer.echo(f"{name:18} {status}  {time.monotonic() - t0:.1f}s  format_errors={retries}")
    raise typer.Exit(1 if failed else 0)
