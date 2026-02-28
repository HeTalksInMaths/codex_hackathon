# Codex Hackathon: Character Degree Verification

Agentic architecture for verifying LLM work on character degrees of groups.

## Overview

This project uses:
- **GAP** for computing character degree sets (signal/verification)
- **Aristotle** for generating Lean verification code
- **LLM agents** for reasoning and analysis

## Focus

Constructing Lean proofs to verify the set of sets of distinct character degrees for groups of order n ≤ 100.

## Setup

1. Copy `.env.example` to `.env` and add your API keys
2. [Additional setup steps to be added]

## Architecture

See `agent_architecture_backend.py` for the Python backend architecture and checks.

## CLI Runner

Use `run_harness.py` to execute the backend flow from JSON files.

Examples:

```bash
# 1) Precheck only (no oracle)
python run_harness.py precheck \
  --targets targets.json \
  --solver solver.json \
  --solver-style BY_GROUP_ID

# 2) Oracle comparison
python run_harness.py compare \
  --targets targets.json \
  --solver solver.json \
  --oracle oracle.json \
  --solver-style BY_GROUP_ID

# 3) Full run + export artifact (precheck + optional compare)
python run_harness.py run \
  --targets targets.json \
  --solver solver.json \
  --oracle oracle.json \
  --solver-style BY_GROUP_ID \
  --out run_artifact.json

# 4) Build LLM prompt from targets
python run_harness.py solver-prompt \
  --targets targets.json \
  --solver-style BY_GROUP_ID \
  --out solver_prompt.txt
```

Notes:
- Exit code is `0` on pass, `2` on benchmark/check failure, `1` on input/runtime error.
- `--spec-lock spec_lock.json` is optional; if omitted, spec lock is generated automatically.

## OpenAI model usage

Current state before this update: the repo had prompt builders/checkers but **no direct OpenAI API call path**, so there was no active model bound in code.

Now you can run an end-to-end model-backed flow with `llm-flow`:

```bash
python run_harness.py llm-flow \
  --targets targets.json \
  --oracle oracle.json \
  --solver-style BY_GROUP_ID \
  --solver-model gpt-5.2 \
  --repair-model gpt-5.2 \
  --harvest-model gpt-5.2 \
  --max-repairs 3 \
  --out llm_run.json
```

Notes:
- Reasoning policy is fixed in flow: solve=`medium`, first repair=`high`, second repair and beyond=`xhigh` alias (sent as API `high`).
- Defaults are configurable with env vars: `OPENAI_MODEL`, `OPENAI_SOLVER_MODEL`, `OPENAI_REPAIR_MODEL`, `OPENAI_HARVEST_MODEL`.
- Requires `OPENAI_API_KEY` in environment.

## Repair loops and planning

This repo now supports two repair-loop helpers:

1) Built-in automatic repair retries inside `llm-flow` via `--max-repairs`.
2) `repair_planner.py` (and `run_harness.py repair-plan`) to generate staged repair plans/prompts from a run artifact.

```bash
python run_harness.py repair-plan --artifact llm_run.json --mode benchmark --out repair_plan.json
python repair_planner.py --artifact llm_run.json --mode benchmark --print-json 1
```
