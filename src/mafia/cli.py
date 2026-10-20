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


@app.command()
def play(run_config: str = O, games: int = 1, view: str = "public", mock: bool = typer.Option(False, hidden=True)):
    """Play games with a run config."""
    from mafia.agents.llm_agent import LLMAgent
    from mafia.config import ROOT, ModelCfg, Personas, Rules, Scoring, load, load_run
    from mafia.engine.game import Game
    from mafia.llm.client import LLMClient
    from mafia.llm.recorder import RecordingLLM, ScriptedLLM
    from mafia.logs.events import EventLog
    from mafia.renderers.text import TextRenderer

    run = load_run(run_config)
    if run.group == "memory":
        typer.echo("memory group not implemented yet")
        raise typer.Exit(1)
    rules, scoring = load(Rules, run.rules_config), load(Scoring, run.scoring_config)
    pcfg = load(Personas, run.personas_config)
    personas = pcfg.personas
    names = {p.agent_id: p.name for p in personas}
    if view not in {"public", "truth", *names}:
        typer.echo(f"unknown view {view!r}; use public, truth or one of: {', '.join(names)}")
        raise typer.Exit(1)
    inner = ScriptedLLM(run.seed, names) if mock else LLMClient(load(ModelCfg, run.model_file))
    agents = {p.agent_id: LLMAgent(p, pcfg, inner, rules, scoring, run.group) for p in personas}
    run_dir = ROOT / "runs" / run.run_id
    for i in range(games):
        llm = RecordingLLM(inner, run_dir / "llm_cache" / f"{i + 1:04d}.jsonl") if run.record_llm_responses else inner
        for a in agents.values():
            a.llm = llm
        g = Game(rules, sorted(personas, key=lambda p: p.seat), agents, run.run_id, i, run.seed, run.group,
                 scoring.version, run.prompt_version)
        g.play()
        EventLog(run_dir, i).write(g.events)
        TextRenderer(names).render_game(g.events, view)


@app.command()
def season(run_config: str = O, resume: bool = False):
    """Run a full season."""
    _todo()


@app.command()
def calibrate(run: str = O):
    """Win rates and proposed scoring_v1.yaml."""
    _todo()


@app.command()
def score(run: str = O, scoring: str = O):
    """Recompute points for a run."""
    _todo()


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
):
    """Inspect an agent's memory."""
    _todo()


@app.command()
def report(runs: list[str] = O):
    """Markdown report for runs."""
    _todo()


@app.command()
def scene(run: str = O, game: int = O, out: str = O):
    """Export a scene script (stub)."""
    _todo()


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
