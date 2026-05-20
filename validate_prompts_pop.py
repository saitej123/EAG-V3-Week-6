#!/usr/bin/env python3
"""
validate_prompts_pop.py

Uses prompt_of_prompts.md (PoP) to evaluate and validate the Perception and Decision
prompts used in the Session 6 agent loop. Outputs the prompts and their JSON evaluations.
Supports concurrent (parallel) evaluation to maximize speed.
"""

import json
import asyncio
import os
import re
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, ConfigDict

from llm_env import gemini_models_ordered, shared_gemini_client

# Define the validation schema matching prompt_of_prompts.md
class PopEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    explicit_reasoning: bool = Field(description="Does the prompt tell the model to reason step-by-step?")
    structured_output: bool = Field(description="Does the prompt enforce a predictable output format (e.g. JSON)?")
    tool_separation: bool = Field(description="Are reasoning steps clearly separated from computation or tool-use?")
    conversation_loop: bool = Field(description="Could this prompt work in a multi-turn conversation setting?")
    instructional_framing: bool = Field(description="Are there examples or exact formats to follow?")
    internal_self_checks: bool = Field(description="Does the prompt instruct the model to self-verify intermediate steps?")
    reasoning_type_awareness: bool = Field(description="Does the prompt encourage tagging/identifying the reasoning type?")
    fallbacks: bool = Field(description="Does the prompt specify what to do on uncertainty/tool failure?")
    overall_clarity: str = Field(description="Overall feedback and clarity summary.")

# Exact key order from prompt_of_prompts.md (lines 45-55)
POP_EVAL_KEYS = (
    "explicit_reasoning",
    "structured_output",
    "tool_separation",
    "conversation_loop",
    "instructional_framing",
    "internal_self_checks",
    "reasoning_type_awareness",
    "fallbacks",
    "overall_clarity",
)


def normalize_pop_eval(parsed: PopEvaluation) -> dict[str, Any]:
    """Return evaluation dict in the exact schema order from prompt_of_prompts.md."""
    data = parsed.model_dump()
    return {key: data[key] for key in POP_EVAL_KEYS}

def extract_prompt_template(filepath: Path) -> str:
    """Extract the literal prompt f-string from a python file."""
    content = filepath.read_text(encoding="utf-8")
    match = re.search(r'prompt\s*=\s*f"""(.*?)"""', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

async def validate_one_prompt(name: str, p_prompt: str, pop_guidelines: str, client: Any, model_id: str, root: Path) -> dict:
    """Evaluate a single prompt f-string against PoP guidelines."""
    user_prompt = f"""
You are evaluating the following prompt used in our system:

PROMPT UNDER EVALUATION:
\"\"\"
{p_prompt}
\"\"\"

Apply the prompt_of_prompts criteria exactly and output the evaluation JSON.
"""
    from google.genai import types

    def _generate():
        response = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=pop_guidelines,
                response_mime_type="application/json",
                response_schema=PopEvaluation,
                temperature=0.2,
            )
        )
        return (response.text or "").strip()

    # Run blocking SDK call in a separate thread to keep asyncio event loop unblocked
    raw = await asyncio.to_thread(_generate)
    data = json.loads(raw)
    parsed = PopEvaluation.model_validate(data)
    result_dict = normalize_pop_eval(parsed)

    # Save to disk
    eval_file = root / f"pop/{name.lower()}_pop_eval.json"
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    
    return result_dict

async def async_validate_all_prompts() -> dict[str, Any]:
    """Exposes parallel validation for backend app.py consumption."""
    root = Path(__file__).resolve().parent
    pop_guidelines = Path(root / "prompt_of_prompts.md").read_text(encoding="utf-8")
    
    perception_path = root / "perception.py"
    decision_path = root / "decision.py"
    
    perception_prompt = extract_prompt_template(perception_path)
    decision_prompt = extract_prompt_template(decision_path)
    
    if not perception_prompt or not decision_prompt:
        raise ValueError("Could not extract Perception or Decision prompt templates.")
        
    client = shared_gemini_client()
    models = gemini_models_ordered()
    if not client or not models:
        raise ValueError("Gemini client not initialized. Check GEMINI_API_KEY in .env")
        
    model_id = models[0]
    
    # Run both validations concurrently in parallel threads
    results = await asyncio.gather(
        validate_one_prompt("Perception", perception_prompt, pop_guidelines, client, model_id, root),
        validate_one_prompt("Decision", decision_prompt, pop_guidelines, client, model_id, root),
        return_exceptions=True
    )
    
    # Handle results/exceptions
    final_output = {}
    for name, res in zip(["Perception", "Decision"], results):
        if isinstance(res, Exception):
            final_output[name] = {"error": str(res)}
        else:
            final_output[name] = res
            
    return {
        "model_id": model_id,
        "perception_prompt": perception_prompt,
        "decision_prompt": decision_prompt,
        "evaluations": final_output
    }

def main():
    print("Using gemini async validation runner...")
    results = asyncio.run(async_validate_all_prompts())
    print("\n--- Perception Evaluation ---")
    print(json.dumps(results["evaluations"]["Perception"], indent=2))
    print("\n--- Decision Evaluation ---")
    print(json.dumps(results["evaluations"]["Decision"], indent=2))
    print("\nPrompt validations completed concurrently!")

if __name__ == "__main__":
    main()
