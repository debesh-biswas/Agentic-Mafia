import typer

app = typer.Typer(help="LLM agents playing Mafia.", no_args_is_help=True)

O = typer.Option(...)  # required option


def _todo():
    typer.echo("not implemented yet")
    raise typer.Exit(1)


@app.command()
def sim(games: int = 200000, seed: int = 1):
    """Random-bot balance check."""
    _todo()


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
    _todo()


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
