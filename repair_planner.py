#!/usr/bin/env python3
"""Generate repair plans/prompts from harness reports.

Supports precheck/oracle report JSON emitted by `run_harness.py run` or `llm-flow`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import agent_architecture_backend as backend


VERY_HIGH = {"PARSE_ERROR", "N_MISMATCH", "SPEC_LOCK_MISMATCH", "DUPLICATE_IDS", "MISSING_GROUPS", "EXTRA_GROUPS"}
HIGH = {"NON_POSINT_DEGREES", "MISSING_DEGREE_1", "SUMSQ_VIOLATION", "DIVIDES_VIOLATION", "LEN_VS_CONJ_CLASSES", "COUNT1_VS_ABELIANIZATION"}
MEDIUM = {"WRONG_GLOBAL_SET_OF_SETS"}
LOW = {"WRONG_DEGREES", "WRONG_DISTINCT_SET"}


@dataclass
class RepairStep:
    title: str
    signal_strength: str
    action: str
    prompt: Optional[str] = None


@dataclass
class RepairPlan:
    stage: str
    n: int
    gating_status: str
    steps: list[RepairStep]
    warnings: list[str]


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(x: Any) -> str:
    return json.dumps(x, indent=2, sort_keys=True)


def _classify(failures: list[dict[str, Any]]) -> str:
    kinds = {str(f.get("kind", "")) for f in failures if isinstance(f, dict)}
    if kinds & VERY_HIGH:
        return "very_high"
    if kinds & HIGH:
        return "high"
    if kinds & MEDIUM:
        return "medium"
    if kinds & LOW:
        return "low"
    return "medium"


def build_plan(artifact: dict[str, Any], mode: str = "benchmark", leak_oracle: bool = False) -> RepairPlan:
    targets = artifact.get("targets") or {}
    solver = artifact.get("solver") or {}
    precheck = artifact.get("precheck") or {}
    oracle_report = artifact.get("oracle_report") or {}
    spec_lock = artifact.get("spec_lock") or backend.build_spec_lock(int(targets.get("n", 0) or 0), "BY_GROUP_ID")

    n = int(targets.get("n", 0) or 0)
    failures = precheck.get("failures") if isinstance(precheck.get("failures"), list) else []
    signal = _classify(failures)

    steps: list[RepairStep] = []
    warnings: list[str] = []

    if not precheck.get("pass"):
        prompt = backend.build_pre_pass_repair_prompt(spec_lock=spec_lock, targets=targets, solver=solver, precheck=precheck)
        steps.append(
            RepairStep(
                title="Repair precheck failures",
                signal_strength=signal,
                action="Patch solver JSON to satisfy schema + arithmetic constraints, then rerun precheck.",
                prompt=prompt,
            )
        )
        return RepairPlan(
            stage="precheck",
            n=n,
            gating_status="Precheck failing: fix before oracle/lean.",
            steps=steps,
            warnings=warnings,
        )

    if oracle_report and not oracle_report.get("global_set_of_sets_pass"):
        safe_report = {
            "summary": oracle_report.get("summary"),
            "failures": [
                {k: v for k, v in f.items() if k not in {"expected"}}
                for f in oracle_report.get("failures", [])
                if isinstance(f, dict)
            ],
        }
        if mode == "dev" and leak_oracle:
            safe_report["oracle_diff"] = oracle_report.get("derived", {}).get("global_set_diff")
        prompt = "\n".join(
            [
                "You are repairing a solver JSON output.",
                "Return ONLY valid JSON matching the original schema.",
                "Locked spec:",
                _fmt(spec_lock),
                "Targets:",
                _fmt({"n": targets.get("n"), "targets": targets.get("targets")}),
                "Previous solver output:",
                _fmt(solver),
                "Comparison report:",
                _fmt(safe_report),
                "Task: produce a corrected solver output. Preserve schema and n/spec_lock exactly.",
            ]
        )
        steps.append(
            RepairStep(
                title="Repair oracle mismatch",
                signal_strength="medium",
                action="Re-derive answer while preserving all hard constraints.",
                prompt=prompt,
            )
        )
        warnings.append("Benchmark mode should avoid oracle leakage; keep leak_oracle=0.")
        return RepairPlan(
            stage="oracle",
            n=n,
            gating_status="Oracle mismatch: do not proceed to Lean.",
            steps=steps,
            warnings=warnings,
        )

    return RepairPlan(
        stage="lean",
        n=n,
        gating_status="Unlocked: precheck/oracle passed.",
        steps=[
            RepairStep(
                title="Proceed to proof harvest / Lean lane",
                signal_strength="high",
                action="Use post-pass harvest prompt and formalize lemmas incrementally.",
                prompt=None,
            )
        ],
        warnings=[],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair planner for char-degree harness artifacts")
    ap.add_argument("--artifact", required=True, help="Path to JSON from run_harness run/llm-flow")
    ap.add_argument("--mode", choices=["benchmark", "dev"], default="benchmark")
    ap.add_argument("--leak-oracle", type=int, default=0)
    ap.add_argument("--print-json", type=int, default=0)
    args = ap.parse_args()

    raw = _read_json(args.artifact)
    artifact = raw.get("artifact", raw)
    if not isinstance(artifact, dict):
        raise ValueError("artifact payload must be an object")

    plan = build_plan(artifact=artifact, mode=args.mode, leak_oracle=bool(args.leak_oracle))

    print(f"REPAIR PLAN | stage={plan.stage} | n={plan.n}")
    print(f"GATING: {plan.gating_status}")
    for idx, step in enumerate(plan.steps, start=1):
        print(f"\n[{idx}] {step.title} ({step.signal_strength})")
        print(f"action: {step.action}")
        if step.prompt:
            print("--- prompt ---")
            print(step.prompt)
            print("--- end prompt ---")

    if plan.warnings:
        print("\nWarnings:")
        for w in plan.warnings:
            print(f"- {w}")

    if args.print_json:
        print("\nPLAN_JSON")
        print(_fmt(asdict(plan)))


if __name__ == "__main__":
    main()
