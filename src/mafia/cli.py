import typer

app = typer.Typer(help="LLM agents playing Mafia.", no_args_is_help=True)


def _todo():
    typer.echo("not implemented yet")
    raise typer.Exit(1)


@app.command()
def sim(games: int = 200000, seed: int = 1):
    """Random-bot balance check."""
    _todo()


@app.command()
def play(run_config: str, games: int = 1, view: str = "public"):
    """Play games with a run config."""
    _todo()


@app.command()
def season(run_config: str, resume: bool = False):
    """Run a full season."""
    _todo()


@app.command()
def calibrate(run: str):
    """Win rates and proposed scoring_v1.yaml."""
    _todo()


@app.command()
def score(run: str, scoring: str):
    """Recompute points for a run."""
    _todo()


@app.command()
def replay(run: str, game: int, view: str = "public"):
    """Replay a recorded game."""
    _todo()


@app.command()
def inspect(run: str, agent: str):
    """Inspect an agent's memory."""
    _todo()


@app.command()
def report(runs: list[str]):
    """Markdown report for runs."""
    _todo()


@app.command()
def scene(run: str, game: int, out: str):
    """Export a scene script (stub)."""
    _todo()
