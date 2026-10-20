"""DRD 13: season runner. `mafia play` and `mafia season` share this one game loop."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mafia.config import ROOT, Memory, ModelCfg, Personas, Rules, Run, Scoring, load, load_run
from mafia.engine.game import Game
from mafia.engine.seeds import game_seed
from mafia.llm.recorder import RecordingLLM, ScriptedLLM
from mafia.logs.events import EventLog
from mafia.logs.manifest import write_manifest
from mafia.metrics import compute
from mafia.reflection.reflect import build_outcome
from mafia.scoring import leaderboard
from mafia.scoring.score import score
from mafia.types import Phase


class RunExists(Exception):
    pass


def after_game_memory(game, agents, points, history, names) -> None:
    """Memory group, after the score event: leaderboard + concurrent post_game for all agents, then their events
    in seat order (DRD 8.1, 13). history includes this game's points."""
    game.st.phase = Phase.REFLECT
    text, ids = leaderboard.render(history, names), list(game.st.players)  # seat order
    outs = [build_outcome(game.events, a, points, names, text, game.st.game_index + 1) for a in ids]
    with ThreadPoolExecutor(len(ids)) as ex:
        results = list(ex.map(lambda ao: agents[ao[0]].post_game(ao[1]), zip(ids, outs)))
    standings = leaderboard.standings(history)
    for a, specs in zip(ids, results):
        game.emit("leaderboard", f"private:{a}", a, standings=standings, text=text)
        for type, vis, payload in specs:
            game.emit(type, vis, a, None, **payload)


def _default_factory(run, rules, scoring, pcfg, llm, run_dir=None, mock=False):
    from mafia.agents.llm_agent import LLMAgent
    if run.group != "memory":
        return {p.agent_id: LLMAgent(p, pcfg, llm, rules, scoring, run.group) for p in pcfg.personas}
    from mafia.memory.embed import Embedder, FakeEmbedder
    from mafia.memory.store import MemoryStore
    mcfg = load(Memory, run.memory_config)
    emb = FakeEmbedder() if mock else Embedder(mcfg.embedding_model)  # one shared, thread-safe embedder
    (run_dir / "memory").mkdir(parents=True, exist_ok=True)
    return {p.agent_id: LLMAgent(p, pcfg, llm, rules, scoring, run.group,
                                 MemoryStore(run_dir / "memory" / f"{p.agent_id}.sqlite", emb), mcfg)
            for p in pcfg.personas}


class CheckpointMissing(RunExists):
    pass


def _drop_db(path: Path) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def _prune_ckpt(ckpt: Path, keep: int) -> None:
    """Remove leftover .tmp dirs and every generation except `keep`."""
    import shutil
    for d in ckpt.glob("g*"):
        if d.name != f"g{keep:04d}":
            shutil.rmtree(d, ignore_errors=True)


def _restore_memory(run_dir: Path, last: int, ids) -> None:
    """DRD 13: memory only reflects completed games. Roll every agent DB back to checkpoint generation `last`;
    with no completed game the DBs are deleted and reseeded."""
    import os
    import shutil
    mem = run_dir / "memory"
    ckpt = mem / "ckpt"
    if last == 0:
        for f in mem.glob("*.sqlite*"):
            f.unlink()
        shutil.rmtree(ckpt, ignore_errors=True)
        return
    gen = ckpt / f"g{last:04d}"
    missing = [a for a in ids if not (gen / f"{a}.sqlite").exists()]
    if missing:
        raise CheckpointMissing(f"memory checkpoint {gen} is missing or incomplete ({', '.join(missing) or 'all'}); "
                                "cannot resume this run safely")
    mem.mkdir(parents=True, exist_ok=True)
    for a in ids:
        db = mem / f"{a}.sqlite"
        _drop_db(db)
        tmp = mem / f"{a}.restore"
        shutil.copy2(gen / f"{a}.sqlite", tmp)
        os.replace(tmp, db)
    _prune_ckpt(ckpt, last)


def _checkpoint(agents, run_dir: Path, g: int) -> None:
    """Back up every agent DB into ckpt/g<g>.tmp, then rename the directory (atomic). Call before state.json."""
    ckpt = run_dir / "memory" / "ckpt"
    tmp = ckpt / f"g{g:04d}.tmp"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for a, ag in agents.items():
        ag.memory.backup(tmp / f"{a}.sqlite")
    final = ckpt / f"g{g:04d}"
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)


def _park(path: Path) -> None:
    if path.exists():
        path.replace(path.with_suffix(".partial"))  # overwrites an older .partial


def run_season(run_cfg, *, mock=False, agent_factory=None, on_game=None, resume=False, games=None,
               runs_dir: Path = ROOT / "runs"):
    """Run (or resume) a season. Memory-group crash safety (per-game DB checkpoints, restore on resume) applies only
    to the default agent factory; a custom agent_factory manages its own state."""
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
    agents = {}
    ckpt_memory = run.group == "memory" and not agent_factory
    if ckpt_memory:
        _restore_memory(run_dir, last, [p.agent_id for p in personas])
    try:
        agents = agent_factory(run, rules, scoring, pcfg, inner) if agent_factory else \
            _default_factory(run, rules, scoring, pcfg, inner, run_dir, mock)
        names = {p.agent_id: p.name for p in personas}
        history = []  # per-game points, for the leaderboard; rebuilt from logs on resume
        if run.group == "memory":
            history = [next(e.payload["points"] for e in EventLog(run_dir, i).read() if e.type == "score") for i in range(last)]
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
                if hasattr(a, "begin_game"):
                    a.begin_game(g, leaderboard.render(history, names) if history else "")
            t0 = time.monotonic()
            game = Game(rules, personas, agents, run.run_id, g - 1, run.seed, run.group, scoring.version, run.prompt_version)
            game.play()
            seconds = time.monotonic() - t0
            points = score(game.events, scoring)
            game.emit("score", "public", points=points)
            if run.group == "memory":
                history.append(points)
                after_game_memory(game, agents, points, history, names)
            EventLog(run_dir, g - 1).write(game.events)
            grow, arows = compute.rows(game.events, scoring, run.run_id, run.group, g, game_seed(run.seed, g - 1), seconds)
            compute.append(run_dir / "games.csv", compute.GAMES_COLS, [grow])
            for r in arows:
                if getattr(agents[r["agent"]], "memory", None):
                    act = agents[r["agent"]].memory.active()
                    r.update(memories_active=len(act), beliefs_active=sum(m.type == "belief" for m in act),
                             lessons_active=sum(m.type == "lesson" for m in act))
            compute.append(run_dir / "agents.csv", compute.AGENTS_COLS, arows)
            if ckpt_memory:
                _checkpoint(agents, run_dir, g)
            state_path.write_text(json.dumps({"last_completed": g}))
            if ckpt_memory:
                _prune_ckpt(run_dir / "memory" / "ckpt", g)
            if on_game:
                on_game(game.events, g)
        write_manifest(run_dir, run, model, scoring.version, ended=True)
    finally:
        if hasattr(inner, "close"):
            inner.close()
        for a in agents.values():
            if getattr(a, "memory", None):
                a.memory.close()
    return run_dir
