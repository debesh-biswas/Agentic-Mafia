# Mafia Agents

LLM agents with memory, reflection and planning play Mafia hundreds of times and learn to deceive, detect and win, without any model training.

## Folder contents

| Path | What |
|---|---|
| `src/mafia/engine/` | Game rules, phases, discussion scheduler, agent views |
| `src/mafia/agents/` | LLM agent, random agent, persona and prompt assembly |
| `src/mafia/llm/` | LLM client, output schemas, prompt templates, record/replay |
| `src/mafia/memory/`, `src/mafia/reflection/` | Per-agent memory store, retrieval, reflection, compression |
| `src/mafia/runner/` | Season runner with resume, calibration |
| `src/mafia/scoring/`, `src/mafia/metrics/` | Points, leaderboard, CSV metrics, comparison report |
| `src/mafia/renderers/` | Text transcripts, scene-script export |
| `config/` | YAML: rules, personas, model endpoints, memory, scoring, run configs |
| `tests/` | Test suite (`pytest`) |
| `tools/balance_sim.py` | Random-play balance check (`python tools/balance_sim.py 200000`) |

## Getting started

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,memory]"   # memory extra only needed for memory-group runs
.venv/bin/pytest -q
mafia sim --games 200000                                       # balance check, no LLM
mafia llm-check --model-config config/model_local.yaml         # check the model endpoint
mafia play --run-config config/run_local.yaml --games 1        # one local LLM game
mafia season --run-config config/run_local_memory.yaml         # 10-game local memory season
mafia report --runs <memory_run> --runs <control_run>
```

Local model: run GPT-OSS 20B in LM Studio (or `mlx_lm.server`) and match `base_url` and `model` in `config/model_local.yaml`.
API runs: `export OPENROUTER_API_KEY=...` and set the real provider in `config/model_openrouter.yaml`.
