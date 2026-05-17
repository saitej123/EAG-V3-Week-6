"""
Session 6 Perception: ``observe`` returns an ``Observation`` (ordered goals).

LLM emits drafts without stable ids; ``artifact_index`` refers to enumerated MEMORY HITS
that carry ``artifact_id``. The outer loop assigns stable ``Goal.id`` by position.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger
from pydantic import ValidationError

from llm_env import gemini_models_ordered, shared_gemini_client
from schemas import Goal, MemoryItem, Observation, PerceptionGoalDraft, PerceptionLLMResponse


class PerceptionModule:
    def __init__(self) -> None:
        self.state = Observation(goals=[])

    def observe(
        self,
        query: str,
        hits: list[MemoryItem],
        history: list[dict[str, Any]],
        prior_goals: list[Goal],
        run_id: str,
    ) -> Observation:
        hits_with_art: list[tuple[int, MemoryItem]] = [(i, h) for i, h in enumerate(hits) if h.artifact_id]

        hits_lines = []
        for j, (_i, h) in enumerate(hits_with_art):
            hits_lines.append(
                f"  artifact_index={j} memory_index={_i} artifact_id={h.artifact_id!r} descriptor={h.descriptor[:160]!r}"
            )
        hits_block = "\n".join(hits_lines) if hits_lines else "  (no memory hits currently carry artifact handles)"

        prior_lines = "\n".join(
            f"  pos={p}: id={g.id!r} done={g.done} text={g.text!r} attach={g.attach_artifact_id!r}"
            for p, g in enumerate(prior_goals)
        ) or "  (none — first decomposition)"

        hist_txt = json.dumps(history[-16:], indent=2, default=str)[:12000]

        prompt = f"""
You are the Perception module for a Session 6 agent. Maintain an ordered goal list.

USER QUERY:
{query}

RUN ID: {run_id}

PRIOR GOALS (preserve order; same positions unless goals complete):
{prior_lines}

MEMORY HITS WITH ARTIFACTS (use artifact_index ONLY from this list; integers 0..{max(0, len(hits_with_art)-1)}):
{hits_block}

RECENT HISTORY (JSON):
{hist_txt}

RULES:
1. If prior_goals is empty: decompose the query into a short ordered list of imperative goals (each ``text`` one line).
2. If prior_goals is non-empty: output EXACTLY len(prior_goals) goals in the SAME ORDER.
   Update ``done`` when history shows the step satisfied. Done goals stay done.
3. For the first unfinished goal, set ``artifact_index`` ONLY when Decision needs fetched bytes now.
   Use the integer from MEMORY HITS WITH ARTIFACTS. Otherwise null.
4. Never invent artifact handles as strings — only integer artifact_index or null.
5. Preserve semantics of each goal; refine ``text`` lightly if needed but do not drop goals.

Respond as JSON matching the schema (goals: list of {{text, done, artifact_index}}).
"""

        client = shared_gemini_client()
        models = gemini_models_ordered()
        llm_goals: list[PerceptionGoalDraft] = []

        if client is None or not models:
            logger.warning("Perception: Gemini unavailable; heuristic fallback.")
            llm_goals = self._fallback_drafts(query, prior_goals)
        else:
            try:
                from google.genai import types

                for model_id in models:
                    try:
                        response = client.models.generate_content(
                            model=model_id,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=PerceptionLLMResponse,
                                temperature=1.0,
                            ),
                        )
                        raw = (response.text or "").strip()
                        data = json.loads(raw)
                        parsed = PerceptionLLMResponse.model_validate(data)
                        llm_goals = parsed.goals
                        break
                    except (json.JSONDecodeError, ValidationError, Exception) as e:
                        logger.warning(f"Perception model={model_id} failed: {e}")
                if not llm_goals:
                    llm_goals = self._fallback_drafts(query, prior_goals)
            except Exception as e:
                logger.warning(f"Perception failed: {e}")
                llm_goals = self._fallback_drafts(query, prior_goals)

        merged = self._merge_goals(prior_goals, llm_goals, hits_with_art)
        if not merged:
            merged = [
                Goal(
                    id=f"g-{uuid.uuid4().hex[:8]}",
                    text=(query or "").strip()[:800] or "Fulfill the user request using tools.",
                    done=False,
                    attach_artifact_id=None,
                )
            ]
            logger.warning("Perception returned no goals — using single catch-all goal.")
        self.state = Observation(goals=merged)
        return self.state

    def _fallback_drafts(self, query: str, prior: list[Goal]) -> list[PerceptionGoalDraft]:
        if prior:
            return [PerceptionGoalDraft(text=g.text, done=g.done, artifact_index=None) for g in prior]
        q = (query or "").strip()[:800]
        return [PerceptionGoalDraft(text=q or "Fulfill the user request using tools.", done=False)]

    def _merge_goals(
        self,
        prior: list[Goal],
        drafts: list[PerceptionGoalDraft],
        hits_with_art: list[tuple[int, MemoryItem]],
    ) -> list[Goal]:
        if not prior:
            out: list[Goal] = []
            for d in drafts:
                gid = f"g-{uuid.uuid4().hex[:8]}"
                attach = self._resolve_attach(d.artifact_index, hits_with_art)
                out.append(Goal(id=gid, text=d.text, done=d.done, attach_artifact_id=attach))
            return out

        n = len(prior)
        padded = list(drafts[:n])
        while len(padded) < n:
            idx = len(padded)
            padded.append(
                PerceptionGoalDraft(text=prior[idx].text, done=prior[idx].done, artifact_index=None)
            )
        padded = padded[:n]

        merged: list[Goal] = []
        for idx in range(n):
            pr = prior[idx]
            d = padded[idx]
            done = bool(pr.done or d.done)
            attach = self._resolve_attach(d.artifact_index, hits_with_art)
            merged.append(
                Goal(
                    id=pr.id,
                    text=d.text or pr.text,
                    done=done,
                    attach_artifact_id=attach if attach else pr.attach_artifact_id,
                )
            )
        return merged

    def _resolve_attach(
        self,
        artifact_index: int | None,
        hits_with_art: list[tuple[int, MemoryItem]],
    ) -> str | None:
        if artifact_index is None:
            return None
        if artifact_index < 0 or artifact_index >= len(hits_with_art):
            return None
        return hits_with_art[artifact_index][1].artifact_id
