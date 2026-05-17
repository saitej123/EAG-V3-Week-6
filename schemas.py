"""
Pydantic v2 contracts for Session 6 boundaries (Memory, Perception, Decision, Action).

Assignment shapes: MemoryItem, Artifact metadata, Goal, Observation, ToolCall, DecisionOutput.
Legacy commerce catalog types (SQLite) remain for Indian PDP caching.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

# --- Session 6 memory ---------------------------------------------------------

MemoryKind = Literal["fact", "preference", "tool_outcome", "scratchpad"]


class MemoryItem(BaseModel):
    """One durable or episodic row in ``state/memory.json``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    kind: MemoryKind
    keywords: list[str] = Field(default_factory=list)
    descriptor: str = ""
    value: dict[str, Any] = Field(default_factory=dict)
    artifact_id: str | None = None
    source: str = ""
    run_id: str = ""
    goal_id: str | None = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArtifactRecord(BaseModel):
    """Metadata sidecar for content-addressable blobs under ``state/artifacts/``."""

    id: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    source: str = ""
    descriptor: str = ""


class Goal(BaseModel):
    """Planner goal with stable ``id`` assigned by the outer loop (not by the LLM)."""

    id: str
    text: str
    done: bool = False
    attach_artifact_id: str | None = None


class Observation(BaseModel):
    """Perception output: ordered goals. Identity is list position + stable ``Goal.id``."""

    goals: list[Goal] = Field(default_factory=list)

    def all_done(self) -> bool:
        return bool(self.goals) and all(g.done for g in self.goals)

    def next_unfinished(self) -> Goal | None:
        for g in self.goals:
            if not g.done:
                return g
        return None


class ToolCall(BaseModel):
    """Single MCP dispatch contract."""

    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class DecisionOutput(BaseModel):
    """Planner branch: prefer ``tool_call`` when present; otherwise ``answer``."""

    answer: str | None = None
    tool_call: ToolCall | None = None

    def resolved(self) -> tuple[str | None, ToolCall | None]:
        if self.tool_call is not None and str(self.tool_call.name).strip():
            return None, self.tool_call
        if self.answer is not None:
            return self.answer, None
        return None, None

    @property
    def is_answer(self) -> bool:
        a, t = self.resolved()
        return a is not None and t is None


class DecisionLLMFlat(BaseModel):
    """Flat JSON schema for Gemini (avoids some nested ``response_schema`` quirks)."""

    branch: Literal["answer", "tool"]
    answer_text: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, JsonValue] = Field(default_factory=dict)


# --- LLM-facing perception draft (no goal ids; loop merges stable ids) -------

class PerceptionGoalDraft(BaseModel):
    text: str
    done: bool = False
    artifact_index: int | None = Field(
        None,
        description="Index into the enumerated MEMORY HITS list that carry artifact_id; null if none.",
    )


class PerceptionLLMResponse(BaseModel):
    goals: list[PerceptionGoalDraft] = Field(default_factory=list)


# --- LLM-facing memory classification on remember() ---------------------------


class MemoryClassifyLLM(BaseModel):
    kind: MemoryKind
    keywords: list[str] = Field(default_factory=list)
    descriptor: str = ""
    value: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.85


# --- Partial summary (max iterations) ----------------------------------------


class PartialSummaryMarkdown(BaseModel):
    markdown_answer: str


# --- Commerce DB (optional catalog) -------------------------------------------

class CommerceProduct(BaseModel):
    platform: str
    product_name: str
    base_price: float
    net_price: float
    bank_offers_text: str
    url: str


class CachedProductRow(BaseModel):
    url: str
    platform: str | None = None
    product_name: str | None = None
    base_price: float | None = None
    net_price: float | None = None
    bank_offers_text: str | None = None
    scraped_at: str | None = None


# ``PerceptionModule`` historically imported ``PerceptionState``; keep as alias to ``Observation``.
PerceptionState = Observation