"""OpenAI-backed agentic flow runner for the character degree harness.

This module wires the existing prompt builders/precheck/oracle checks to OpenAI
Responses API calls so model choice is explicit and configurable per step.
"""

from __future__ import annotations

import json
import os
import re
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


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def _extract_first_json_span(text: str) -> Optional[str]:
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None

    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _normalize_smart_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _parse_json_lenient(text: str) -> tuple[dict[str, Any], str]:
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []

    def add(name: str, candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append((name, candidate))

    add("strict_raw", text)
    add("trimmed", text.replace("\ufeff", "").strip())

    for name, candidate in list(candidates):
        fenced = _strip_code_fences(candidate)
        if fenced != candidate:
            add(f"{name}+strip_code_fence", fenced)

    for name, candidate in list(candidates):
        extracted = _extract_first_json_span(candidate)
        if extracted and extracted != candidate:
            add(f"{name}+extract_json_span", extracted)

    for name, candidate in list(candidates):
        smart = _normalize_smart_quotes(candidate)
        if smart != candidate:
            add(f"{name}+normalize_smart_quotes", smart)

    for name, candidate in list(candidates):
        no_trailing = _remove_trailing_commas(candidate)
        if no_trailing != candidate:
            add(f"{name}+remove_trailing_commas", no_trailing)

    last_err: Optional[json.JSONDecodeError] = None
    for name, candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
        if not isinstance(parsed, dict):
            raise LLMFlowError(f"Model JSON root must be an object; got {type(parsed).__name__}")
        return parsed, name

    if last_err is None:
        raise LLMFlowError("Model returned empty/invalid output after normalization attempts")
    raise LLMFlowError(f"Model did not return valid JSON: {last_err}\nRaw output:\n{text}")


def _call_json_model(client: OpenAI, prompt: str, model: str, reasoning_effort: str) -> tuple[dict[str, Any], str, str]:
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
    payload, parse_strategy = _parse_json_lenient(text)
    return payload, effort, parse_strategy


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
    solver, sent_effort, parse_strategy = _call_json_model(client, solver_prompt, cfg.solver_model, "medium")
    call_trace.append(
        {
            "step": "solve",
            "model": cfg.solver_model,
            "effort_requested": "medium",
            "effort_sent": sent_effort,
            "parse_strategy": parse_strategy,
        }
    )

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
        solver, sent_effort, parse_strategy = _call_json_model(client, repair_prompt, cfg.repair_model, requested_effort)
        call_trace.append(
            {
                "step": f"repair_{repair_attempts + 1}",
                "model": cfg.repair_model,
                "effort_requested": requested_effort,
                "effort_sent": sent_effort,
                "parse_strategy": parse_strategy,
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
            proof_plan, sent_effort, parse_strategy = _call_json_model(client, harvest_prompt, cfg.harvest_model, "high")
            call_trace.append(
                {
                    "step": "harvest",
                    "model": cfg.harvest_model,
                    "effort_requested": "high",
                    "effort_sent": sent_effort,
                    "parse_strategy": parse_strategy,
                }
            )

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
