#!/usr/bin/env python3
"""
Evaluate an LLM on pathway_analysis_env cases.

Runs the agent loop on each case file, records whether the LLM
identifies the correct pathway, and prints a summary table.

Usage:
    export OPENAI_API_KEY=sk-...
    PYTHONPATH=src:envs python3 examples/pathway_eval.py

    # Different model:
    MODEL=gpt-4o PYTHONPATH=src:envs python3 examples/pathway_eval.py

    # Different provider (HuggingFace, Ollama, etc.):
    API_BASE_URL=http://localhost:11434/v1 MODEL=llama3 API_KEY=x \
        PYTHONPATH=src:envs python3 examples/pathway_eval.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))

from pathway_analysis_env.models import PathwayAction, PathwayObservation
from pathway_analysis_env.server.pathway_environment import PathwayEnvironment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_STEPS = 8

CASES = [
    "toy_case_001.json",
    "toy_case_002.json",
    "toy_case_legacy.json",
]

SYSTEM_PROMPT = """\
You are a computational biology research agent. You have access to a \
synthetic omics dataset and must identify which signaling pathway is \
activated.

Available actions (reply with exactly ONE JSON object per turn):

1. {"action_type": "inspect_dataset"}
   View dataset metadata and available experimental conditions.

2. {"action_type": "run_differential_expression"}
   Run differential expression analysis to find top DE genes.

3. {"action_type": "run_pathway_enrichment"}
   Run pathway enrichment to find top enriched pathways.

4. {"action_type": "submit_answer", "hypothesis": "<pathway name>"}
   Submit your final answer. Use the exact pathway name from the \
   enrichment results.

Strategy: inspect the dataset first, then run DE and enrichment to \
gather evidence, then submit your answer. Reply with ONLY the JSON \
action object, no other text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def observation_to_text(obs: PathwayObservation) -> str:
    parts = [f"Message: {obs.message}"]
    if obs.available_conditions:
        parts.append(f"Available conditions: {obs.available_conditions}")
    if obs.top_genes:
        parts.append(f"Top DE genes: {obs.top_genes}")
    if obs.top_pathways:
        parts.append(f"Top enriched pathways: {obs.top_pathways}")
    if obs.metadata:
        parts.append(f"Metadata: {json.dumps(obs.metadata)}")
    return "\n".join(parts)


def parse_action(text: str) -> Optional[PathwayAction]:
    json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None
    if not data.get("action_type"):
        return None
    return PathwayAction(
        action_type=data["action_type"],
        condition_a=data.get("condition_a"),
        condition_b=data.get("condition_b"),
        gene_list=data.get("gene_list"),
        hypothesis=data.get("hypothesis"),
    )


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------


def run_episode(
    case_file: str,
    client: OpenAI,
    verbose: bool = False,
) -> Dict[str, Any]:
    env = PathwayEnvironment(case_file=case_file)
    obs = env.reset()
    true_pathway = env.state.true_pathway

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Environment observation:\n{observation_to_text(obs)}"},
    ]

    actions_taken: List[str] = []
    total_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=256,
            temperature=0.0,
        )
        assistant_text = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": assistant_text})

        action = parse_action(assistant_text)
        if action is None:
            history.append({
                "role": "user",
                "content": "Could not parse your response. Reply with a single JSON object.",
            })
            continue

        actions_taken.append(action.action_type)
        if verbose:
            extra = f" -> {action.hypothesis}" if action.hypothesis else ""
            print(f"  Step {step}: {action.action_type}{extra}")

        obs = env.step(action)
        total_reward += float(obs.reward or 0)

        if obs.done:
            correct = obs.metadata.get("correct", False)
            submitted = action.hypothesis or ""
            return {
                "case": case_file,
                "true_pathway": true_pathway,
                "submitted": submitted,
                "correct": correct,
                "reward": total_reward,
                "steps": step,
                "actions": actions_taken,
            }

        history.append({
            "role": "user",
            "content": f"Environment observation:\n{observation_to_text(obs)}",
        })

    return {
        "case": case_file,
        "true_pathway": true_pathway,
        "submitted": "",
        "correct": False,
        "reward": total_reward,
        "steps": MAX_STEPS,
        "actions": actions_taken,
        "error": "max_steps_reached",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        print("Set OPENAI_API_KEY (or API_KEY + API_BASE_URL) to run eval.")
        raise SystemExit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"Model:  {MODEL}")
    print(f"API:    {API_BASE_URL}")
    print(f"Cases:  {len(CASES)}")
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for case_file in CASES:
        print(f"\n>> {case_file}")
        result = run_episode(case_file, client, verbose=True)
        results.append(result)
        mark = "PASS" if result["correct"] else "FAIL"
        print(f"   {mark} | submitted: {result['submitted']!r} | true: {result['true_pathway']!r} | reward: {result['reward']}")

    elapsed = time.time() - t0

    # Summary
    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    avg_reward = sum(r["reward"] for r in results) / total
    avg_steps = sum(r["steps"] for r in results) / total

    print()
    print("=" * 70)
    print(f"RESULTS  {correct_count}/{total} correct  ({correct_count/total:.0%})")
    print(f"         avg reward: {avg_reward:.2f}  avg steps: {avg_steps:.1f}  time: {elapsed:.1f}s")
    print("=" * 70)

    print()
    print(f"{'Case':<30} {'Correct':>8} {'Submitted':<25} {'Reward':>7} {'Steps':>6}")
    print("-" * 80)
    for r in results:
        mark = "yes" if r["correct"] else "NO"
        print(f"{r['case']:<30} {mark:>8} {r['submitted']:<25} {r['reward']:>7.1f} {r['steps']:>6}")


if __name__ == "__main__":
    main()
