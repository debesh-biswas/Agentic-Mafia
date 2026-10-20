"""DRD 11.3 manifest.json."""
import hashlib
import json
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import yaml

from mafia.config import ROOT, Run


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def prompts_hash() -> str:
    d = ROOT / "src/mafia/llm/prompts"
    h = hashlib.sha256()
    for f in sorted(d.glob("*.txt")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


class InputsChanged(Exception):
    pass


def snapshot_inputs(run_dir: Path, run: Run) -> None:
    """Copy the configs a run uses into <run_dir>/config/ on first start; on resume refuse if any config file or the
    prompt templates differ from what the run started with (DRD 4.3)."""
    cdir = Path(run_dir) / "config"
    files = {"run.yaml": yaml.safe_dump(run.model_dump(mode="json", by_alias=True))}
    for name, src in (("model", run.model_file), ("scoring", run.scoring_config), ("rules", run.rules_config),
                      ("personas", run.personas_config), ("memory", run.memory_config)):
        files[f"{name}.yaml"] = Path(src).read_text()
    man = Path(run_dir) / "manifest.json"
    if cdir.exists():
        def same(n, t):  # `games` may be raised on resume to extend a season
            if not (cdir / n).exists():
                return False
            a, b = ((yaml.safe_load(x) if n == "run.yaml" else x) for x in ((cdir / n).read_text(), t))
            if n == "run.yaml":
                a.pop("games", None), b.pop("games", None)
            return a == b
        bad = [n for n, t in files.items() if not same(n, t)]
        old = json.loads(man.read_text()).get("prompts_sha256") if man.exists() else None
        if old and old != prompts_hash():
            bad.append("prompts")
        if bad:
            raise InputsChanged(f"cannot resume {run.run_id}: changed since the run started: {', '.join(bad)}")
        return
    cdir.mkdir(parents=True)
    for n, t in files.items():
        (cdir / n).write_text(t)


def write_manifest(run_dir: Path, run: Run, model: dict, scoring_version: int, ended: bool = False) -> None:
    """First call stamps `start`; later calls keep it. `end` is set when ended=True."""
    path = Path(run_dir) / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({
        "run_id": run.run_id, "run_config": run.model_dump(mode="json"), "git_commit": _git_commit(),
        "git_dirty": old["git_dirty"] if "git_dirty" in old else _git_dirty(),
        "prompts_sha256": old.get("prompts_sha256") or prompts_hash(),
        "package_version": version("mafia"), "start": old.get("start") or datetime.now().isoformat(),
        "end": datetime.now().isoformat() if ended else None, "model": model.get("model"),
        "provider_pin": model.get("provider_pin"), "scoring_version": scoring_version, "prompt_version": run.prompt_version,
        "seed": run.seed, "group": run.group}, indent=2))
