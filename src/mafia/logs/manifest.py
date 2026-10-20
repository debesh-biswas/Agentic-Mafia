"""DRD 11.3 manifest.json."""
import json
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from mafia.config import ROOT, Run


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def write_manifest(run_dir: Path, run: Run, model: dict, scoring_version: int, ended: bool = False) -> None:
    """First call stamps `start`; later calls keep it. `end` is set when ended=True."""
    path = Path(run_dir) / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({
        "run_id": run.run_id, "run_config": run.model_dump(mode="json"), "git_commit": _git_commit(),
        "package_version": version("mafia"), "start": old.get("start") or datetime.now().isoformat(),
        "end": datetime.now().isoformat() if ended else None, "model": model.get("model"),
        "provider": model.get("provider"), "scoring_version": scoring_version, "prompt_version": run.prompt_version,
        "seed": run.seed, "group": run.group}, indent=2))
