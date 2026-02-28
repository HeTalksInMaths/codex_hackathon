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
