#!/usr/bin/env python3
"""CLI runner for the character-degree harness backend.

This wraps `agent_architecture_backend.py` so you can run the pipeline from files.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Any, Optional

import agent_architecture_backend as backend
import llm_flow
import repair_planner


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _read_json(path: str) -> Any:
    return json.loads(_read_text(path))


def _write_json(payload: Any, out_path: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out_path:
        Path(out_path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _load_spec_lock(spec_lock_path: Optional[str], n: int, solver_style: backend.SolverStyle) -> dict[str, Any]:
    if spec_lock_path:
        spec_lock = _read_json(spec_lock_path)
        if not isinstance(spec_lock, dict):
            raise ValueError("spec_lock file must contain a JSON object")
        return spec_lock
    return backend.build_spec_lock(n=n, solver_style=solver_style)


def cmd_precheck(args: argparse.Namespace) -> int:
    targets = backend.parse_targets_json(_read_text(args.targets))
    solver = backend.parse_solver_json(_read_text(args.solver), solver_style=args.solver_style)
    spec_lock = _load_spec_lock(args.spec_lock, n=int(targets["n"]), solver_style=args.solver_style)

    report = backend.run_precheck(
        targets_obj=targets,
        solver_obj=solver,
        solver_style=args.solver_style,
        spec_lock=spec_lock,
    )
    _write_json(report, args.out)
    return 0 if report.get("pass") else 2


def cmd_compare(args: argparse.Namespace) -> int:
    targets = backend.parse_targets_json(_read_text(args.targets))
    solver = backend.parse_solver_json(_read_text(args.solver), solver_style=args.solver_style)
    oracle = backend.parse_oracle_json(_read_text(args.oracle))
    spec_lock = _load_spec_lock(args.spec_lock, n=int(targets["n"]), solver_style=args.solver_style)

    report = backend.run_oracle_comparison(
        targets_obj=targets,
        solver_obj=solver,
        oracle_obj=oracle,
        solver_style=args.solver_style,
        spec_lock=spec_lock,
    )
    _write_json(report, args.out)
    return 0 if report.get("global_set_of_sets_pass") else 2


def cmd_run(args: argparse.Namespace) -> int:
    targets = backend.parse_targets_json(_read_text(args.targets))
    solver = backend.parse_solver_json(_read_text(args.solver), solver_style=args.solver_style)
    spec_lock = _load_spec_lock(args.spec_lock, n=int(targets["n"]), solver_style=args.solver_style)

    precheck = backend.run_precheck(
        targets_obj=targets,
        solver_obj=solver,
        solver_style=args.solver_style,
        spec_lock=spec_lock,
    )

    oracle_obj = None
    oracle_report = None
    if args.oracle:
        oracle_obj = backend.parse_oracle_json(_read_text(args.oracle))
        oracle_report = backend.run_oracle_comparison(
            targets_obj=targets,
            solver_obj=solver,
            oracle_obj=oracle_obj,
            solver_style=args.solver_style,
            spec_lock=spec_lock,
        )

    artifact = backend.build_export_artifact(
        n=int(targets["n"]),
        solver_style=args.solver_style,
        lean_unlock_policy=args.lean_unlock_policy,
        spec_lock=spec_lock,
        targets=targets,
        solver=solver,
        oracle=oracle_obj,
        precheck=precheck,
        oracle_report=oracle_report,
    )

    result = {
        "spec_lock": spec_lock,
        "precheck": precheck,
        "oracle_report": oracle_report,
        "artifact": artifact,
    }
    _write_json(result, args.out)

    if args.oracle:
        return 0 if (precheck.get("pass") and oracle_report and oracle_report.get("global_set_of_sets_pass")) else 2
    return 0 if precheck.get("pass") else 2


def cmd_solver_prompt(args: argparse.Namespace) -> int:
    targets = backend.parse_targets_json(_read_text(args.targets))
    spec_lock = _load_spec_lock(args.spec_lock, n=int(targets["n"]), solver_style=args.solver_style)
    prompt = backend.build_solver_prompt_v2(targets=targets, spec_lock=spec_lock, solver_style=args.solver_style)
    if args.out:
        Path(args.out).write_text(prompt + "\n", encoding="utf-8")
    else:
        print(prompt)
    return 0


def cmd_repair_prompt(args: argparse.Namespace) -> int:
    targets = backend.parse_targets_json(_read_text(args.targets))
    solver = backend.parse_solver_json(_read_text(args.solver), solver_style=args.solver_style)
    precheck = _read_json(args.precheck)
    if not isinstance(precheck, dict):
        raise ValueError("precheck file must contain a JSON object")
    spec_lock = _load_spec_lock(args.spec_lock, n=int(targets["n"]), solver_style=args.solver_style)

    prompt = backend.build_pre_pass_repair_prompt(
        spec_lock=spec_lock,
        targets=targets,
        solver=solver,
        precheck=precheck,
    )
    if args.out:
        Path(args.out).write_text(prompt + "\n", encoding="utf-8")
    else:
        print(prompt)
    return 0


def cmd_llm_flow(args: argparse.Namespace) -> int:
    targets = backend.parse_targets_json(_read_text(args.targets))
    oracle = backend.parse_oracle_json(_read_text(args.oracle)) if args.oracle else None
    spec_lock = _load_spec_lock(args.spec_lock, n=int(targets["n"]), solver_style=args.solver_style)

    model_config = llm_flow.ModelConfig(
        solver_model=args.solver_model,
        repair_model=args.repair_model,
        harvest_model=args.harvest_model,
    )

    result = llm_flow.run_agentic_flow(
        targets=targets,
        solver_style=args.solver_style,
        spec_lock=spec_lock,
        oracle=oracle,
        unlock_policy=args.lean_unlock_policy,
        model_config=model_config,
        max_repairs=args.max_repairs,
    )
    _write_json(result, args.out)

    pre_ok = bool(result.get("precheck", {}).get("pass"))
    if args.oracle:
        oracle_ok = bool(result.get("oracle_report", {}).get("global_set_of_sets_pass"))
        return 0 if pre_ok and oracle_ok else 2
    return 0 if pre_ok else 2


def cmd_repair_plan(args: argparse.Namespace) -> int:
    payload = _read_json(args.artifact)
    artifact = payload.get("artifact", payload)
    if not isinstance(artifact, dict):
        raise ValueError("artifact JSON must be an object or contain an artifact object")

    plan = repair_planner.build_plan(
        artifact=artifact,
        mode=args.mode,
        leak_oracle=bool(args.leak_oracle),
    )
    _write_json(asdict(plan), args.out)
    return 0


def _add_common_io_args(p: argparse.ArgumentParser, include_oracle: bool = False) -> None:
    p.add_argument("--targets", required=True, help="Path to targets JSON")
    p.add_argument("--solver", required=True, help="Path to solver JSON")
    if include_oracle:
        p.add_argument("--oracle", required=True, help="Path to oracle JSON")
    p.add_argument(
        "--solver-style",
        required=True,
        choices=["BY_GROUP_ID", "GLOBAL_SET_OF_SETS"],
        help="Solver output style",
    )
    p.add_argument("--spec-lock", help="Optional path to spec_lock JSON (defaults to auto-generated)")
    p.add_argument("--out", help="Optional output path (defaults to stdout)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Character Degree Harness CLI (file-based runner)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_precheck = sub.add_parser("precheck", help="Run precheck (no oracle)")
    _add_common_io_args(p_precheck, include_oracle=False)
    p_precheck.set_defaults(func=cmd_precheck)

    p_compare = sub.add_parser("compare", help="Run oracle comparison")
    _add_common_io_args(p_compare, include_oracle=True)
    p_compare.set_defaults(func=cmd_compare)

    p_run = sub.add_parser("run", help="Run precheck and optional oracle comparison + artifact")
    _add_common_io_args(p_run, include_oracle=False)
    p_run.add_argument("--oracle", help="Optional oracle JSON path; if set, comparison is executed")
    p_run.add_argument(
        "--lean-unlock-policy",
        default="GLOBAL_SET_OF_SETS",
        choices=["STRICT_BY_ID", "GLOBAL_SET_OF_SETS"],
        help="Included in export artifact",
    )
    p_run.set_defaults(func=cmd_run)

    p_solver_prompt = sub.add_parser("solver-prompt", help="Build solver prompt from targets")
    p_solver_prompt.add_argument("--targets", required=True, help="Path to targets JSON")
    p_solver_prompt.add_argument(
        "--solver-style",
        required=True,
        choices=["BY_GROUP_ID", "GLOBAL_SET_OF_SETS"],
        help="Solver output style",
    )
    p_solver_prompt.add_argument("--spec-lock", help="Optional path to spec_lock JSON")
    p_solver_prompt.add_argument("--out", help="Optional output path (defaults to stdout)")
    p_solver_prompt.set_defaults(func=cmd_solver_prompt)

    p_repair_prompt = sub.add_parser("repair-prompt", help="Build repair prompt from precheck report")
    p_repair_prompt.add_argument("--targets", required=True, help="Path to targets JSON")
    p_repair_prompt.add_argument("--solver", required=True, help="Path to solver JSON")
    p_repair_prompt.add_argument("--precheck", required=True, help="Path to precheck JSON")
    p_repair_prompt.add_argument(
        "--solver-style",
        required=True,
        choices=["BY_GROUP_ID", "GLOBAL_SET_OF_SETS"],
        help="Solver output style",
    )
    p_repair_prompt.add_argument("--spec-lock", help="Optional path to spec_lock JSON")
    p_repair_prompt.add_argument("--out", help="Optional output path (defaults to stdout)")
    p_repair_prompt.set_defaults(func=cmd_repair_prompt)

    p_llm_flow = sub.add_parser("llm-flow", help="Run full LLM-backed agentic flow (solve, optional repair, optional harvest)")
    p_llm_flow.add_argument("--targets", required=True, help="Path to targets JSON")
    p_llm_flow.add_argument("--oracle", help="Optional oracle JSON path")
    p_llm_flow.add_argument(
        "--solver-style",
        required=True,
        choices=["BY_GROUP_ID", "GLOBAL_SET_OF_SETS"],
        help="Solver output style",
    )
    p_llm_flow.add_argument("--spec-lock", help="Optional path to spec_lock JSON")
    p_llm_flow.add_argument(
        "--lean-unlock-policy",
        default="GLOBAL_SET_OF_SETS",
        choices=["STRICT_BY_ID", "GLOBAL_SET_OF_SETS"],
        help="Included in export artifact",
    )
    p_llm_flow.add_argument("--solver-model", default=llm_flow.DEFAULT_MODEL, help="OpenAI model used for initial solve")
    p_llm_flow.add_argument("--repair-model", default=llm_flow.DEFAULT_MODEL, help="OpenAI model used for repair pass")
    p_llm_flow.add_argument("--harvest-model", default=llm_flow.DEFAULT_MODEL, help="OpenAI model used for proof harvest")
    p_llm_flow.add_argument("--max-repairs", type=int, default=2, help="Maximum automated repair attempts after initial solve")
    p_llm_flow.add_argument("--out", help="Optional output path (defaults to stdout)")
    p_llm_flow.set_defaults(func=cmd_llm_flow)

    p_repair_plan = sub.add_parser("repair-plan", help="Build repair plan/prompts from run artifact JSON")
    p_repair_plan.add_argument("--artifact", required=True, help="Path to run artifact JSON")
    p_repair_plan.add_argument("--mode", choices=["benchmark", "dev"], default="benchmark")
    p_repair_plan.add_argument("--leak-oracle", type=int, default=0, help="Dev only: include oracle diff hints")
    p_repair_plan.add_argument("--out", help="Optional output path (defaults to stdout)")
    p_repair_plan.set_defaults(func=cmd_repair_plan)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
