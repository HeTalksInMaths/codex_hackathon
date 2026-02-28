"""OpenAI-backed agentic flow runner for the character degree harness.

This module wires the existing prompt builders/precheck/oracle checks to OpenAI
Responses API calls so model choice is explicit and configurable per step.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

import agent_architecture_backend as backend


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")


@dataclass
class ModelConfig:
    solver_model: str = os.getenv("OPENAI_SOLVER_MODEL", DEFAULT_MODEL)
    repair_model: str = os.getenv("OPENAI_REPAIR_MODEL", DEFAULT_MODEL)
    harvest_model: str = os.getenv("OPENAI_HARVEST_MODEL", DEFAULT_MODEL)


class LLMFlowError(RuntimeError):
    pass


def normalize_reasoning_effort(reasoning_effort: str) -> str:
    effort = (reasoning_effort or "high").lower().strip()
    if effort == "xhigh":
        return "high"
    if effort not in {"low", "medium", "high"}:
        raise ValueError("reasoning effort must be one of: low, medium, high, xhigh")
    return effort


def _extract_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    chunks: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) in {"output_text", "text"}:
                maybe = getattr(content, "text", None)
                if maybe:
                    chunks.append(maybe)
    return "\n".join(chunks).strip()


def _call_json_model(client: OpenAI, prompt: str, model: str, reasoning_effort: str) -> tuple[dict[str, Any], str]:
    effort = normalize_reasoning_effort(reasoning_effort)
    resp = client.responses.create(
        model=model,
        reasoning={"effort": effort},
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    )
    text = _extract_text(resp)
    if not text:
        raise LLMFlowError("Model returned empty response")
    try:
        return json.loads(text), effort
    except json.JSONDecodeError as exc:
        raise LLMFlowError(f"Model did not return valid JSON: {exc}\nRaw output:\n{text}") from exc


def _repair_effort(repair_attempt_index: int) -> str:
    if repair_attempt_index <= 0:
        return "high"
    return "xhigh"


def run_agentic_flow(
    *,
    targets: dict[str, Any],
    solver_style: backend.SolverStyle,
    spec_lock: dict[str, Any],
    oracle: Optional[dict[str, Any]] = None,
    unlock_policy: backend.UnlockPolicy = "GLOBAL_SET_OF_SETS",
    model_config: Optional[ModelConfig] = None,
    max_repairs: int = 2,
) -> dict[str, Any]:
    client = OpenAI()
    cfg = model_config or ModelConfig()

    call_trace: list[dict[str, str]] = []

    solver_prompt = backend.build_solver_prompt_v2(targets=targets, spec_lock=spec_lock, solver_style=solver_style)
    solver, sent_effort = _call_json_model(client, solver_prompt, cfg.solver_model, "medium")
    call_trace.append({"step": "solve", "model": cfg.solver_model, "effort_requested": "medium", "effort_sent": sent_effort})

    precheck = backend.run_precheck(
        targets_obj=targets,
        solver_obj=solver,
        solver_style=solver_style,
        spec_lock=spec_lock,
    )

    repair_attempts = 0
    while not precheck.get("pass") and repair_attempts < max(0, int(max_repairs)):
        repair_prompt = backend.build_pre_pass_repair_prompt(
            spec_lock=spec_lock,
            targets=targets,
            solver=solver,
            precheck=precheck,
        )
        requested_effort = _repair_effort(repair_attempts)
        solver, sent_effort = _call_json_model(client, repair_prompt, cfg.repair_model, requested_effort)
        call_trace.append(
            {
                "step": f"repair_{repair_attempts + 1}",
                "model": cfg.repair_model,
                "effort_requested": requested_effort,
                "effort_sent": sent_effort,
            }
        )
        precheck = backend.run_precheck(
            targets_obj=targets,
            solver_obj=solver,
            solver_style=solver_style,
            spec_lock=spec_lock,
        )
        repair_attempts += 1

    oracle_report = None
    proof_plan = None
    if oracle is not None:
        oracle_report = backend.run_oracle_comparison(
            targets_obj=targets,
            solver_obj=solver,
            oracle_obj=oracle,
            solver_style=solver_style,
            spec_lock=spec_lock,
        )
        if oracle_report.get("global_set_of_sets_pass"):
            harvest_prompt = backend.build_post_pass_proof_harvest_prompt(
                n=int(targets["n"]),
                targets=targets,
                solver_style=solver_style,
                solver_obj=solver,
                unlock_policy=unlock_policy,
                oracle_report=oracle_report,
            )
            proof_plan, sent_effort = _call_json_model(client, harvest_prompt, cfg.harvest_model, "high")
            call_trace.append({"step": "harvest", "model": cfg.harvest_model, "effort_requested": "high", "effort_sent": sent_effort})

    artifact = backend.build_export_artifact(
        n=int(targets["n"]),
        solver_style=solver_style,
        lean_unlock_policy=unlock_policy,
        spec_lock=spec_lock,
        targets=targets,
        solver=solver,
        oracle=oracle,
        precheck=precheck,
        oracle_report=oracle_report,
    )

    return {
        "model_config": {
            "solver_model": cfg.solver_model,
            "repair_model": cfg.repair_model,
            "harvest_model": cfg.harvest_model,
        },
        "reasoning_policy": {
            "solve": "medium",
            "repair_1": "high",
            "repair_2_plus": "xhigh",
            "harvest": "high",
        },
        "call_trace": call_trace,
        "repair_attempts": repair_attempts,
        "solver": solver,
        "precheck": precheck,
        "oracle_report": oracle_report,
        "proof_plan": proof_plan,
        "artifact": artifact,
    }
