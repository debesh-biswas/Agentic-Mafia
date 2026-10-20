from pathlib import Path
from typing import Literal, TypeVar

import yaml
from mafia.types import Phase, Role
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]  # repo root (editable install)


class Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class Rules(Cfg):
    players: int
    roles: dict[Role, int]
    start_phase: Phase
    max_day: int
    speech_max_words: int
    last_words_max_words: int
    notes_max_words: int
    plan_max_words: int
    turns_per_player_per_day: int
    max_turns_per_player_per_day: int
    priority_pass_consumes_turn: bool
    urgency_min: int
    urgency_max: int
    doctor_self_protect_consecutive: bool
    allow_abstain: bool
    allow_self_vote: bool
    reveal_roles_on_elimination: bool


CALL_TYPES = {"plan", "night", "speak", "vote", "last_words", "reflect", "summaries", "deep_reflect", "compress"}


class ModelCfg(Cfg):
    name: str
    base_url: str
    api_key_env: str | None
    model: str
    provider_pin: dict | None = None  # OpenRouter only
    structured_output: Literal["json_schema", "json_object", "prompt_only"]
    reasoning_param: str
    reasoning: dict[str, str]
    temperature: float
    max_output_tokens: dict[str, int]
    timeout_s: int
    max_concurrency: int

    @model_validator(mode="after")
    def _check(self):
        missing = CALL_TYPES - self.reasoning.keys()
        if missing:
            raise ValueError(f"reasoning missing call types: {sorted(missing)}")
        if self.provider_pin is not None and not self.provider_pin.get("order"):
            raise ValueError("provider_pin needs a non-empty 'order'")
        if self.provider_pin and any("<" in str(x) or ">" in str(x) for x in self.provider_pin["order"]):
            raise ValueError("provider_pin.order still has a placeholder; set the real provider name before running")
        return self


class Retrieval(Cfg):
    top_k: int
    weights: dict[str, float]
    recency_decay_per_game: float
    include_failed_lessons: int


class Reflection(Cfg):
    questions: int
    memories_per_question: int
    max_insights: int


class Compression(Cfg):
    every_games: int = Field(gt=0)
    merge_episodic_older_than_games: int = Field(ge=0)


class Fading(Cfg):
    max_importance: int
    unretrieved_for_games: int = Field(ge=0)


class Memory(Cfg):
    embedding_model: str
    retrieval: Retrieval
    related_beliefs_for_reflection: int
    reflection: Reflection
    compression: Compression
    fading: Fading
    deep_reflection_every_games: int = Field(gt=0)


class Scoring(Cfg):
    version: int
    win_points: dict[str, int]
    bonus: dict[str, int]


class Persona(Cfg):
    agent_id: str
    name: str
    age: int
    seat: int
    background: str
    voice: str
    examples: list[str]
    how_you_play: str
    talk_level: str
    social_stance: str
    risk: str
    play_style: str
    update_style: str
    self_insight: str
    closest_friend: str | None
    private_line_about_closest: str | None
    appearance: dict


class Personas(Cfg):
    group_impressions: dict[str, str]
    personas: list[Persona]


class Run(Cfg):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), populate_by_name=True)

    run_id: str
    group: Literal["memory", "control"]
    games: int
    seed: int
    model_file: Path = Field(alias="model_config")  # YAML key clashes with pydantic's model_config
    scoring_config: Path
    rules_config: Path
    personas_config: Path
    memory_config: Path
    prompt_version: int
    record_llm_responses: bool
    snapshot_memory_every: int = Field(gt=0)


T = TypeVar("T", bound=Cfg)


def load(model: type[T], path: str | Path) -> T:
    return model.model_validate(yaml.safe_load(Path(path).read_text()))


def load_run(path: str | Path) -> Run:
    """Load a run config; referenced config paths are resolved against the repo root."""
    run = load(Run, path)
    for f in ("model_file", "scoring_config", "rules_config", "personas_config", "memory_config"):
        setattr(run, f, ROOT / getattr(run, f))
    return run
