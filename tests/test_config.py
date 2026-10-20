from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mafia.config import ROOT, Memory, ModelCfg, Personas, Rules, Run, Scoring, load, load_run

C = ROOT / "config"


@pytest.mark.parametrize("model,name", [
    (Rules, "rules"), (ModelCfg, "model_local"), (ModelCfg, "model_openrouter"),
    (Memory, "memory"), (Scoring, "scoring_v0"), (Personas, "personas"),
])
def test_loads(model, name):
    load(model, C / f"{name}.yaml")


def test_personas_seven():
    assert len(load(Personas, C / "personas.yaml").personas) == 7


@pytest.mark.parametrize("name", ["run_local", "run_calibration", "run_memory", "run_control"])
def test_runs(name):
    r = load_run(C / f"{name}.yaml")
    assert r.rules_config == ROOT / "config/rules.yaml"


def test_unknown_key_rejected(tmp_path):
    d = yaml.safe_load((C / "rules.yaml").read_text()) | {"bogus": 1}
    p = tmp_path / "r.yaml"
    p.write_text(yaml.dump(d))
    with pytest.raises(ValidationError):
        load(Rules, p)
