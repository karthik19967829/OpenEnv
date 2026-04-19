#!/usr/bin/env python3
"""Solve a pathway analysis task with an LLM.

This script runs the PathwayEnvironment directly (no server/Docker needed)
and asks an OpenAI-compatible model to identify the activated signaling
pathway by choosing which analysis actions to run and then submitting a
hypothesis.

Prerequisites
-------------
1. Install the environment (from the repo root)::

       pip install -e .
       pip install openai

2. Set your API key::

       export OPENAI_API_KEY=sk-...
       # or for HuggingFace router / vLLM / Ollama:
       export API_KEY=your_key
       export API_BASE_URL=https://router.huggingface.co/v1

3. Run::

       PYTHONPATH=src:envs python examples/pathway_analysis_inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
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
VERBOSE = True

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
    """Convert a PathwayObservation into a human-readable string for the LLM."""
    parts = [f"Message: {obs.message}"]
    if obs.available_conditions:
        parts.append(f"Available conditions: {obs.available_conditions}")
    if obs.top_genes:
        parts.append(f"Top DE genes: {obs.top_genes}")
    if obs.top_pathways:
        parts.append(f"Top enriched pathways: {obs.top_pathways}")
    if obs.metadata:
        parts.append(f"Metadata: {json.dumps(obs.metadata)}")
    if obs.reward is not None:
        parts.append(f"Reward: {obs.reward}")
    return "\n".join(parts)


def parse_action(text: str) -> Optional[PathwayAction]:
    """Extract a PathwayAction from the LLM's response text."""
    json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    action_type = data.get("action_type")
    if not action_type:
        return None

    return PathwayAction(
        action_type=action_type,
        condition_a=data.get("condition_a"),
        condition_b=data.get("condition_b"),
        gene_list=data.get("gene_list"),
        hypothesis=data.get("hypothesis"),
    )


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_episode(
    env: PathwayEnvironment,
    client: OpenAI,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Run one full episode: LLM chooses actions until done or max steps."""

    obs = env.reset()
    obs_text = observation_to_text(obs)

    history: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Environment observation:\n{obs_text}"},
    ]

    transcript: List[Dict[str, Any]] = []

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=256,
            temperature=0.0,
        )

        assistant_text = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": assistant_text})

        if VERBOSE:
            print(f"\n--- Step {step} ---")
            print(f"LLM response: {assistant_text}")

        action = parse_action(assistant_text)
        if action is None:
            feedback = (
                "Could not parse a valid action from your response. "
                "Please reply with a single JSON object like "
                '{"action_type": "inspect_dataset"}'
            )
            history.append({"role": "user", "content": feedback})
            transcript.append({"step": step, "error": "parse_failure", "raw": assistant_text})
            continue

        if VERBOSE:
            print(f"Action: {action.action_type}", end="")
            if action.hypothesis:
                print(f" (hypothesis: {action.hypothesis})", end="")
            print()

        obs = env.step(action)
        obs_text = observation_to_text(obs)

        if VERBOSE:
            print(f"Observation: {obs.message}")
            if obs.top_genes:
                print(f"  Genes: {obs.top_genes}")
            if obs.top_pathways:
                print(f"  Pathways: {obs.top_pathways}")
            print(f"  Reward: {obs.reward}  Done: {obs.done}")

        transcript.append({
            "step": step,
            "action": action.action_type,
            "hypothesis": action.hypothesis,
            "reward": obs.reward,
            "done": obs.done,
            "correct": obs.metadata.get("correct"),
        })

        if obs.done:
            correct = obs.metadata.get("correct", False)
            return correct, transcript

        history.append(
            {"role": "user", "content": f"Environment observation:\n{obs_text}"}
        )

    return False, transcript


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        print("Error: Set OPENAI_API_KEY or API_KEY to query the model.")
        print()
        print("Examples:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  export API_KEY=your_key API_BASE_URL=https://router.huggingface.co/v1")
        raise SystemExit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = PathwayEnvironment()

    print(f"Model: {MODEL}")
    print(f"API:   {API_BASE_URL}")
    print(f"Case:  toy_case_001 (true pathway: MAPK signaling)")
    print("=" * 60)

    correct, transcript = run_episode(env, client)

    print()
    print("=" * 60)
    if correct:
        print("RESULT: Agent correctly identified the pathway!")
    else:
        print("RESULT: Agent did NOT identify the correct pathway.")

    total_reward = sum(t.get("reward", 0) or 0 for t in transcript)
    print(f"Total reward: {total_reward}")
    print(f"Steps taken: {len(transcript)}")

    print("\n--- Full transcript ---")
    for entry in transcript:
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
