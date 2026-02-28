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
            lean_code_parts.append(f"-- axiom {lemma_name} : True  -- TODO: Formalize this lemma")
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


def count_actual_sorries(lean_code: str) -> int:
    """Count only actual sorry statements, not in comments."""
    count = 0
    for line in lean_code.split('\n'):
        stripped = line.strip()
        # Count lines that are just sorry or end with sorry (not in comments)
        if not stripped.startswith('--') and not stripped.startswith('/-'):
            if stripped == 'sorry' or (stripped.endswith('sorry') and not ':=' in stripped):
                count += 1
    return count


def extract_theorems_from_skeleton(lean_code: str) -> tuple[str, list[dict[str, str]]]:
    """Extract individual theorems from a Lean skeleton.

    Args:
        lean_code: Full Lean skeleton with imports and multiple theorems

    Returns:
        Tuple of (header with imports, list of theorem dicts)
        Each theorem dict has: {'section': '## Case N', 'code': 'theorem ... sorry'}
    """
    lines = lean_code.split('\n')

    # Find where the first case/theorem starts
    header_lines = []
    theorem_blocks = []
    current_block = None
    current_section = None
    in_lemma_backlog = False

    for line in lines:
        # Check if we hit the lemma backlog section
        if '/-! ## Required Lemmas' in line:
            in_lemma_backlog = True
            continue

        # Check for case section marker
        if line.startswith('/-! ## Case'):
            # Save previous block if any
            if current_block is not None:
                theorem_blocks.append({
                    'section': current_section,
                    'code': '\n'.join(current_block)
                })
            # Start new block
            current_section = line.strip()
            current_block = [line]
        elif current_block is not None:
            # We're in a theorem block
            current_block.append(line)
        elif not in_lemma_backlog and not current_block:
            # Still in header
            header_lines.append(line)

    # Save last block
    if current_block is not None:
        theorem_blocks.append({
            'section': current_section,
            'code': '\n'.join(current_block)
        })

    header = '\n'.join(header_lines).rstrip() + '\n\n'
    return header, theorem_blocks


async def call_aristotle_for_theorem_async(
    header: str,
    theorem_code: str,
    theorem_name: str,
    timeout: int = 600,
    api_key: Optional[str] = None
) -> tuple[str, dict[str, Any]]:
    """Call Aristotle to complete a single theorem.

    Args:
        header: Lean imports and file header
        theorem_code: Single theorem block with sorry
        theorem_name: Name for logging (e.g., "Case 1")
        timeout: Timeout in seconds
        api_key: Aristotle API key

    Returns:
        Tuple of (completed_code, metadata)
    """
    # Combine header + theorem for a standalone file
    lean_file = header + theorem_code

    if api_key is None:
        api_key = os.getenv("ARISTOTLE_API_KEY")
        if not api_key:
            raise ValueError("ARISTOTLE_API_KEY not found in environment")

    try:
        api_request.set_api_key(api_key)

        print(f"  Calling Aristotle for {theorem_name}...")

        # Save to temp file
        temp_path = f"_aristotle_temp_{os.getpid()}_{theorem_name.replace(' ', '_')}.lean"
        Path(temp_path).write_text(lean_file, encoding='utf-8')

        try:
            result = await project.Project.prove_from_file(
                input_file_path=temp_path,
                wait_for_completion=True,
                polling_interval_seconds=10,
            )

            # Read result
            if isinstance(result, str) and os.path.exists(result):
                solution_content = Path(result).read_text(encoding='utf-8')
                os.unlink(result)
            else:
                solution_content = result if isinstance(result, str) else lean_file

            # Extract just the theorem part (remove header)
            completed_theorem = solution_content[len(header):] if solution_content.startswith(header.split('\n')[0]) else solution_content

            metadata = {
                "status": "success",
                "theorem": theorem_name,
                "original_sorries": count_actual_sorries(theorem_code),
                "completed_sorries": count_actual_sorries(completed_theorem),
            }

            return completed_theorem, metadata

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        metadata = {
            "status": "error",
            "theorem": theorem_name,
            "error": str(e),
        }
        return theorem_code, metadata


def integrate_aristotle_into_flow_iterative(
    llm_run_result: dict[str, Any],
    output_lean_path: str,
    aristotle_timeout: int = 600,
) -> dict[str, Any]:
    """Process theorems one at a time with Aristotle (faster feedback).

    Args:
        llm_run_result: Output from llm_flow.run_agentic_flow()
        output_lean_path: Where to save completed Lean file
        aristotle_timeout: Timeout per theorem in seconds

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

    # Extract individual theorems
    header, theorem_blocks = extract_theorems_from_skeleton(lean_skeleton)
    print(f"Found {len(theorem_blocks)} theorems to process")

    # Process each theorem
    print("\n🤖 Processing theorems one at a time with Aristotle...")
    completed_theorems = []
    all_metadata = []

    async def process_all():
        for idx, block in enumerate(theorem_blocks, 1):
            section = block['section']
            theorem_code = block['code']

            print(f"\n[{idx}/{len(theorem_blocks)}] {section}")

            completed, meta = await call_aristotle_for_theorem_async(
                header=header,
                theorem_code=theorem_code,
                theorem_name=f"Case {idx}",
                timeout=aristotle_timeout
            )

            completed_theorems.append(completed)
            all_metadata.append(meta)

            if meta.get("status") == "success":
                orig = meta.get("original_sorries", 0)
                comp = meta.get("completed_sorries", 0)
                if comp == 0:
                    print(f"  ✅ Completed! ({orig} sorries resolved)")
                elif comp < orig:
                    print(f"  ⚠️  Partial: {comp}/{orig} sorries remaining")
                else:
                    print(f"  ⚠️  No progress: {comp} sorries still present")
            else:
                print(f"  ❌ Error: {meta.get('error', 'Unknown')}")

    # Run async processing
    asyncio.run(process_all())

    # Combine results
    final_code = header + '\n'.join(completed_theorems)

    # Add back the lemma backlog section
    backlog_start = lean_skeleton.find('/-! ## Required Lemmas')
    if backlog_start >= 0:
        final_code += '\n\n' + lean_skeleton[backlog_start:]

    # Save completed code
    Path(output_lean_path).write_text(final_code, encoding="utf-8")
    print(f"\n✅ Saved completed Lean: {output_lean_path}")

    # Calculate totals
    total_original_sorries = sum(m.get("original_sorries", 0) for m in all_metadata)
    total_completed_sorries = sum(m.get("completed_sorries", 0) for m in all_metadata)
    success_count = sum(1 for m in all_metadata if m.get("status") == "success")

    return {
        "status": "success" if success_count == len(theorem_blocks) else "partial",
        "skeleton_path": skeleton_path,
        "completed_path": output_lean_path,
        "total_theorems": len(theorem_blocks),
        "successful_theorems": success_count,
        "original_sorries": total_original_sorries,
        "completed_sorries": total_completed_sorries,
        "per_theorem_metadata": all_metadata,
    }


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
