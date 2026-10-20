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
                 record=bool(save_run))
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
def play(run_config: str = O, games: int = 1, view: str = "public"):
    """Play games with a run config."""
    _todo()


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
    TextRenderer(names).render_game(EventLog(ROOT / "runs" / run, game - 1).read(), view)


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
