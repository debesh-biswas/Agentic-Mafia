"""DRD 13: season runner. `mafia play` and `mafia season` share this one game loop."""
import json
import time
from pathlib import Path

from mafia.config import ROOT, ModelCfg, Personas, Rules, Run, Scoring, load, load_run
from mafia.engine.game import Game
from mafia.engine.seeds import game_seed
from mafia.llm.recorder import RecordingLLM, ScriptedLLM
from mafia.logs.events import EventLog
from mafia.logs.manifest import write_manifest
from mafia.metrics import compute
from mafia.scoring.score import score


class RunExists(Exception):
    pass


def after_game_memory(run, agents, events, points, game_no) -> None:
    """Memory-group hooks (leaderboard event, post_game, deep reflect, compress, snapshot). No-op until step 7."""


def _default_factory(run, rules, scoring, pcfg, llm):
    from mafia.agents.llm_agent import LLMAgent
    return {p.agent_id: LLMAgent(p, pcfg, llm, rules, scoring, run.group) for p in pcfg.personas}


def _park(path: Path) -> None:
    if path.exists():
        path.replace(path.with_suffix(".partial"))  # overwrites an older .partial


def run_season(run_cfg, *, mock=False, agent_factory=None, on_game=None, resume=False, games=None,
               runs_dir: Path = ROOT / "runs"):
    run: Run = load_run(run_cfg) if isinstance(run_cfg, (str, Path)) else run_cfg
    total = games or run.games
    rules, scoring = load(Rules, run.rules_config), load(Scoring, run.scoring_config)
    pcfg = load(Personas, run.personas_config)
    personas = sorted(pcfg.personas, key=lambda p: p.seat)
    run_dir = Path(runs_dir) / run.run_id
    if run_dir.exists() and not resume:
        raise RunExists(f"{run_dir} already exists; pass --resume to continue it or delete it first")
    state_path = run_dir / "state.json"
    last = json.loads(state_path.read_text())["last_completed"] if state_path.exists() else 0
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("games.csv", "agents.csv"):
        compute.truncate(run_dir / f, last)
    if mock:
        inner, model = ScriptedLLM(run.seed, {p.agent_id: p.name for p in personas}), {"model": "mock"}
    else:
        from mafia.llm.client import LLMClient
        mcfg = load(ModelCfg, run.model_file)
        inner, model = LLMClient(mcfg), {"model": mcfg.model, "provider": (mcfg.provider_pin or {}).get("order")}
    write_manifest(run_dir, run, model, scoring.version)
    try:
        agents = (agent_factory or _default_factory)(run, rules, scoring, pcfg, inner)
        for g in range(last + 1, total + 1):
            _park(EventLog(run_dir, g - 1).path)  # a log without state.json is a crashed game
            llm = inner
            if run.record_llm_responses:
                cache = run_dir / "llm_cache" / f"{g:04d}.jsonl"
                _park(cache)
                llm = RecordingLLM(inner, cache)
            for a in agents.values():
                if hasattr(a, "llm"):
                    a.llm = llm
            t0 = time.monotonic()
            game = Game(rules, personas, agents, run.run_id, g - 1, run.seed, run.group, scoring.version, run.prompt_version)
            game.play()
            seconds = time.monotonic() - t0
            points = score(game.events, scoring)
            game.emit("score", "public", points=points)
            EventLog(run_dir, g - 1).write(game.events)
            grow, arows = compute.rows(game.events, scoring, run.run_id, run.group, g, game_seed(run.seed, g - 1), seconds)
            compute.append(run_dir / "games.csv", compute.GAMES_COLS, [grow])
            compute.append(run_dir / "agents.csv", compute.AGENTS_COLS, arows)
            if run.group == "memory":
                after_game_memory(run, agents, game.events, points, g)
            state_path.write_text(json.dumps({"last_completed": g}))
            if on_game:
                on_game(game.events, g)
        write_manifest(run_dir, run, model, scoring.version, ended=True)
    finally:
        if hasattr(inner, "close"):
            inner.close()
    return run_dir
