"""Aristotle API integration for completing Lean proofs.

This module takes a proof plan and generates Lean code with sorry statements,
then uses Aristotle to complete the proofs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import asyncio
from dotenv import load_dotenv
from aristotlelib import api_request, project

# Load .env file
load_dotenv()


def build_lean_skeleton_from_proof_plan(proof_plan: dict[str, Any], n: int) -> str:
    """Convert a proof plan JSON into Lean skeleton with sorry statements.

    Args:
        proof_plan: The proof_plan from llm_flow harvest step
        n: Group order

    Returns:
        Lean code with theorems containing sorry placeholders
    """

    lean_code_parts = [
        "import Mathlib.GroupTheory.GroupAction.Basic",
        "import Mathlib.GroupTheory.SpecificGroups.Cyclic",
        "import Mathlib.RepresentationTheory.Basic",
        "import Mathlib.Data.Fintype.Card",
        "",
        "/-!",
        f"# Character Degrees for Groups of Order {n}",
        "",
        f"This file formalizes the proof that groups of order {n} have specific",
        "character degree sets.",
        "-/",
        "",
    ]

    degree_set_plans = proof_plan.get("degree_set_plans", [])

    for idx, plan in enumerate(degree_set_plans, start=1):
        distinct_degrees = plan.get("distinct_degrees", [])
        strategy = plan.get("strategy", "")
        steps = plan.get("steps", [])

        # Generate theorem statement
        degrees_str = "{" + ", ".join(map(str, distinct_degrees)) + "}"
        theorem_name = f"character_degrees_n{n}_case{idx}"

        lean_code_parts.append(f"/-! ## Case {idx}: Degrees {degrees_str} -/")
        lean_code_parts.append("")
        lean_code_parts.append(f"/-- Strategy: {strategy} -/")
        lean_code_parts.append(f"theorem {theorem_name}")
        lean_code_parts.append(f"  (G : Type*) [Group G] [Fintype G]")
        lean_code_parts.append(f"  (h_card : Fintype.card G = {n}) :")
        lean_code_parts.append(f"  ∃ (χs : List ℂ), χs.toFinset = {degrees_str} := by")

        # Add proof steps as comments
        for step_idx, step in enumerate(steps, start=1):
            kind = step.get("kind", "")
            claim = step.get("claim", "")
            lean_lemma = step.get("lean_lemma_needed", "")

            lean_code_parts.append(f"  -- Step {step_idx} (kind {kind}): {claim}")
            if lean_lemma:
                lean_code_parts.append(f"  -- Requires: {lean_lemma}")

        lean_code_parts.append("  sorry")
        lean_code_parts.append("")

    # Add lemma backlog as commented stubs
    lean_backlog = proof_plan.get("lean_backlog", [])
    if lean_backlog:
        lean_code_parts.append("/-! ## Required Lemmas (backlog) -/")
        lean_code_parts.append("")

        for lemma in lean_backlog:
            lemma_name = lemma.get("lemma_name", "")
            statement = lemma.get("statement_in_english", "")
            priority = lemma.get("priority", "P2")

            lean_code_parts.append(f"/-- Priority {priority}: {statement} -/")
            lean_code_parts.append(f"-- lemma {lemma_name} : sorry := sorry")
            lean_code_parts.append("")

    return "\n".join(lean_code_parts)


async def call_aristotle_api_async(
    lean_code: str,
    timeout: int = 600,
    api_key: Optional[str] = None
) -> tuple[str, dict[str, Any]]:
    """Call Aristotle API to complete sorry statements in Lean code (async).

    Args:
        lean_code: Lean source code with sorry statements
        timeout: Timeout in seconds (default 10 minutes)
        api_key: Aristotle API key (defaults to ARISTOTLE_API_KEY env var)

    Returns:
        Tuple of (completed_lean_code, metadata)
    """

    if api_key is None:
        api_key = os.getenv("ARISTOTLE_API_KEY")
        if not api_key:
            raise ValueError("ARISTOTLE_API_KEY not found in environment")

    try:
        # Set API key
        api_request.set_api_key(api_key)

        print(f"Calling Aristotle API (timeout: {timeout}s)...")
        print("This may take 5-15 minutes for complex proofs...")

        # Save to temp file in current directory (needs lakefile.lean and lean-toolchain)
        temp_path = f"_aristotle_temp_{os.getpid()}.lean"
        Path(temp_path).write_text(lean_code, encoding='utf-8')

        try:
            # Call Aristotle to complete the proof using file path
            # Returns the path to the completed file, not the content
            result = await project.Project.prove_from_file(
                input_file_path=temp_path,
                wait_for_completion=True,
                polling_interval_seconds=15,
            )

            # Aristotle returns a file path - read the content
            if isinstance(result, str):
                # Result is a file path
                solution_path = result
                if os.path.exists(solution_path):
                    solution_content = Path(solution_path).read_text(encoding='utf-8')
                    # Clean up Aristotle's temp file
                    os.unlink(solution_path)
                else:
                    solution_content = lean_code
            else:
                # Result is content directly
                solution_content = result

            metadata = {
                "status": "success",
                "original_sorries": lean_code.count("sorry"),
                "completed_sorries": solution_content.count("sorry"),
                "timeout": timeout,
            }

            return solution_content, metadata

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        metadata = {
            "status": "error",
            "error": str(e),
            "timeout": timeout,
        }
        return lean_code, metadata


def call_aristotle_api(
    lean_code: str,
    timeout: int = 600,
    api_key: Optional[str] = None
) -> tuple[str, dict[str, Any]]:
    """Synchronous wrapper for Aristotle API call."""
    return asyncio.run(call_aristotle_api_async(lean_code, timeout, api_key))


def integrate_aristotle_into_flow(
    llm_run_result: dict[str, Any],
    output_lean_path: str,
    aristotle_timeout: int = 600,
) -> dict[str, Any]:
    """Take LLM flow results and complete proofs with Aristotle.

    Args:
        llm_run_result: Output from llm_flow.run_agentic_flow()
        output_lean_path: Where to save completed Lean file
        aristotle_timeout: Timeout for Aristotle API call

    Returns:
        Dictionary with aristotle results and metadata
    """

    # Check if we have a proof plan
    proof_plan = llm_run_result.get("proof_plan")
    if not proof_plan:
        return {
            "status": "skipped",
            "reason": "No proof plan available (oracle check may have failed)"
        }

    # Get n from artifact
    artifact = llm_run_result.get("artifact", {})
    n = artifact.get("n", 0)
    if not n:
        return {
            "status": "error",
            "reason": "Could not determine n from artifact"
        }

    # Build Lean skeleton
    print(f"\nGenerating Lean skeleton for n={n}...")
    lean_skeleton = build_lean_skeleton_from_proof_plan(proof_plan, n)

    # Save skeleton
    skeleton_path = output_lean_path.replace(".lean", "_skeleton.lean")
    Path(skeleton_path).write_text(lean_skeleton, encoding="utf-8")
    print(f"Saved skeleton: {skeleton_path}")
    print(f"Sorries to complete: {lean_skeleton.count('sorry')}")

    # Call Aristotle
    print("\n🤖 Calling Aristotle API to complete proofs...")
    completed_code, metadata = call_aristotle_api(
        lean_skeleton,
        timeout=aristotle_timeout
    )

    # Save completed code
    Path(output_lean_path).write_text(completed_code, encoding="utf-8")
    print(f"\n✅ Saved completed Lean: {output_lean_path}")

    return {
        "status": metadata.get("status"),
        "skeleton_path": skeleton_path,
        "completed_path": output_lean_path,
        "original_sorries": lean_skeleton.count("sorry"),
        "completed_sorries": completed_code.count("sorry"),
        "metadata": metadata,
    }


if __name__ == "__main__":
    # Test skeleton generation
    test_plan = {
        "n": 6,
        "degree_set_plans": [
            {
                "distinct_degrees": [1],
                "strategy": "Use abelian theorem",
                "steps": [
                    {"kind": "A", "claim": "G is abelian"},
                    {"kind": "B", "claim": "Abelian => linear chars", "lean_lemma_needed": "abelian_imp_linear"}
                ]
            }
        ],
        "lean_backlog": []
    }

    skeleton = build_lean_skeleton_from_proof_plan(test_plan, 6)
    print(skeleton)
