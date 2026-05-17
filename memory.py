"""
Session 6 Memory service: typed ``MemoryItem`` rows in ``state/memory.json``.

Reads are keyword-ranked (no LLM). ``remember`` runs one structured classification LLM call.
``record_outcome`` appends episodic tool rows without an LLM.

Commerce SQLite catalog remains alongside episodic memory for PDP caching.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from llm_env import gemini_models_ordered, shared_gemini_client
from schemas import CachedProductRow, CommerceProduct, MemoryClassifyLLM, MemoryItem, MemoryKind, ToolCall

_PROJECT_ROOT = Path(__file__).resolve().parent
STATE_DIR = _PROJECT_ROOT / "state"
MEMORY_JSON_PATH = STATE_DIR / "memory.json"
DB_PATH = STATE_DIR / "commerce.db"

_STOPWORDS = frozenset(
    """
    a an the and or but if to of in on for with as by at from into through during before after above below
    between under again further then once here there when where why how all both each few more most other some
    such no nor not only own same so than too very can will just don should now is are was were be been being
    it its this that these those my your our their me him her them us i you he she we they what which who whom
    """.split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"\W+", text.lower()) if t and t not in _STOPWORDS and len(t) > 1}


def _memory_value_dict_from_json_blob(raw: str) -> dict[str, Any]:
    """Parse ``MemoryClassifyLLM.value_json`` into ``MemoryItem.value`` (Developer API cannot use map schemas)."""
    s = (raw or "").strip() or "{}"
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return {"value": obj}
    except json.JSONDecodeError:
        return {"text": s}


class MemoryService:
    def __init__(self) -> None:
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"MemoryService: cannot create state dir: {e}")
            raise
        self._items: list[MemoryItem] = []
        self._load_disk()
        self._init_db()

    # -- persistence ---------------------------------------------------------

    def _load_disk(self) -> None:
        if not MEMORY_JSON_PATH.exists():
            self._items = []
            self._save_disk()
            return
        try:
            raw = json.loads(MEMORY_JSON_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"memory.json unreadable ({e}); starting empty.")
            self._items = []
            return

        items: list[MemoryItem] = []
        if isinstance(raw, dict) and isinstance(raw.get("items"), list):
            for row in raw["items"]:
                try:
                    items.append(MemoryItem.model_validate(row))
                except ValidationError:
                    continue
        elif isinstance(raw, dict) and isinstance(raw.get("facts"), list):
            for f in raw["facts"]:
                if not isinstance(f, dict):
                    continue
                text = str(f.get("text", "")).strip()
                kws = [str(x).lower() for x in f.get("keywords", []) if x]
                items.append(
                    MemoryItem(
                        id=f"mig-{uuid.uuid4().hex[:12]}",
                        kind="fact",
                        keywords=kws,
                        descriptor=text[:240] or "(fact)",
                        value={"text": text},
                        artifact_id=None,
                        source="migrated_facts_v1",
                        run_id="",
                        goal_id=None,
                        confidence=1.0,
                        created_at=datetime.now(timezone.utc),
                    )
                )
        self._items = items

    def _save_disk(self) -> None:
        try:
            MEMORY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {"items": [m.model_dump(mode="json") for m in self._items]}
            MEMORY_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as e:
            logger.error(f"memory.json write failed: {e}")

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS products (
                        url TEXT PRIMARY KEY,
                        platform TEXT,
                        product_name TEXT,
                        base_price REAL,
                        net_price REAL,
                        bank_offers_text TEXT,
                        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQLite init failed ({DB_PATH}): {e}")

    # -- Session 6 API -------------------------------------------------------

    def read(
        self,
        query: str,
        history: list[dict[str, Any]],
        *,
        kinds: list[MemoryKind] | None = None,
        top_k: int = 8,
    ) -> list[MemoryItem]:
        """Keyword-ranked recall (no LLM)."""
        if kinds:
            pool = [m for m in self._items if m.kind in kinds]
        else:
            pool = list(self._items)

        ctx_bits: list[str] = []
        for h in history[-24:]:
            try:
                ctx_bits.append(json.dumps(h, default=str))
            except Exception:
                ctx_bits.append(str(h))
        ctx = " ".join(ctx_bits)
        qt = _tokens(query + " " + ctx)

        def score(m: MemoryItem) -> float:
            desc_t = _tokens(m.descriptor)
            key_t = set(m.keywords)
            return float(len(qt & desc_t) + len(qt & key_t) + 0.25 * len(qt & _tokens(json.dumps(m.value))))

        ranked = sorted(pool, key=score, reverse=True)
        return ranked[: max(1, top_k)]

    def filter(
        self,
        *,
        kinds: list[MemoryKind] | None = None,
        goal_id: str | None = None,
        recent: int | None = None,
    ) -> list[MemoryItem]:
        out = list(self._items)
        if kinds:
            out = [m for m in out if m.kind in kinds]
        if goal_id:
            out = [m for m in out if m.goal_id == goal_id]
        out.sort(key=lambda m: m.created_at, reverse=True)
        if recent is not None:
            out = out[:recent]
        return out

    def remember(
        self,
        raw_text: str,
        *,
        source: str,
        run_id: str,
        goal_id: str | None = None,
    ) -> None:
        """Classify free-form text via one structured LLM call, then persist."""
        text = (raw_text or "").strip()
        if not text:
            return

        classified = self._classify_with_llm(text)
        item = MemoryItem(
            id=f"mem-{uuid.uuid4().hex[:12]}",
            kind=classified.kind,
            keywords=classified.keywords,
            descriptor=classified.descriptor or text[:240],
            value=_memory_value_dict_from_json_blob(classified.value_json),
            artifact_id=None,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
            confidence=classified.confidence,
            created_at=datetime.now(timezone.utc),
        )
        self._items.append(item)
        self._save_disk()
        logger.info(
            f"[memory.remember] kind={item.kind} keywords={item.keywords[:8]} descriptor={item.descriptor[:120]!r}"
        )

    def _classify_with_llm(self, text: str) -> MemoryClassifyLLM:
        client = shared_gemini_client()
        models = gemini_models_ordered()
        prompt = f"""
Classify the following user content for a durable agent memory store.

Return JSON matching the schema with:
- kind: one of fact | preference | tool_outcome | scratchpad (use "fact" for birthdays and stated truths).
- keywords: short lowercase tokens useful for keyword recall.
- descriptor: ONE short human-readable line.
- value_json: ONE JSON **object** serialized as a string with canonical fields when obvious (e.g. {{"entity":"…","date":"…"}}).

Content:
{text}
"""
        if client is None or not models:
            return MemoryClassifyLLM(
                kind="fact",
                keywords=[w for w in _tokens(text)][:12],
                descriptor=text[:200],
                value_json=json.dumps({"text": text}, ensure_ascii=False),
                confidence=0.5,
            )

        try:
            from google.genai import types

            last_err: Exception | None = None
            for model_id in models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=MemoryClassifyLLM,
                            temperature=1.0,
                        ),
                    )
                    raw = (response.text or "").strip()
                    data = json.loads(raw)
                    return MemoryClassifyLLM.model_validate(data)
                except Exception as e:
                    last_err = e
                    logger.warning(f"[memory.classify] model={model_id} failed: {e}")
            logger.warning(f"[memory.classify] fallback heuristic after {last_err!r}")
        except Exception as e:
            logger.warning(f"[memory.classify] failed {e}")

        return MemoryClassifyLLM(
            kind="fact",
            keywords=[w for w in _tokens(text)][:12],
            descriptor=text[:200],
            value_json=json.dumps({"text": text}, ensure_ascii=False),
            confidence=0.5,
        )

    def record_outcome(
        self,
        *,
        tool_call: ToolCall,
        result_text: str,
        artifact_id: str | None,
        run_id: str,
        goal_id: str | None,
    ) -> MemoryItem:
        desc = f"{tool_call.name}({json.dumps(tool_call.arguments, default=str)[:180]}) → artifact={artifact_id}"
        item = MemoryItem(
            id=f"out-{uuid.uuid4().hex[:12]}",
            kind="tool_outcome",
            keywords=[tool_call.name.lower()]
            + [w for w in _tokens(json.dumps(tool_call.arguments, default=str))][:8],
            descriptor=desc[:500],
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "preview": result_text[:4000],
            },
            artifact_id=artifact_id,
            source="mcp",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
            created_at=datetime.now(timezone.utc),
        )
        self._items.append(item)
        self._save_disk()
        return item

    # -- commerce catalog (existing assignment / concierge path) --------------

    def upsert_product(self, product: CommerceProduct) -> None:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO products (url, platform, product_name, base_price, net_price, bank_offers_text, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(url) DO UPDATE SET
                        platform=excluded.platform,
                        product_name=excluded.product_name,
                        base_price=excluded.base_price,
                        net_price=excluded.net_price,
                        bank_offers_text=excluded.bank_offers_text,
                        scraped_at=CURRENT_TIMESTAMP
                    """,
                    (
                        product.url,
                        product.platform,
                        product.product_name,
                        product.base_price,
                        product.net_price,
                        product.bank_offers_text,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"upsert_product failed: {e}")

    def query_products(self, search_term: str = "") -> list[CachedProductRow]:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = "SELECT url, platform, product_name, base_price, net_price, bank_offers_text, scraped_at FROM products"
                params: list[Any] = []
                if search_term:
                    query += " WHERE product_name LIKE ? OR url LIKE ? OR platform LIKE ?"
                    pat = f"%{search_term}%"
                    params.extend([pat, pat, pat])
                query += " ORDER BY scraped_at DESC LIMIT 100"
                cursor.execute(query, params)
                rows = cursor.fetchall()
                out: list[CachedProductRow] = []
                for row in rows:
                    try:
                        out.append(CachedProductRow.model_validate(dict(row)))
                    except ValidationError:
                        continue
                return out
        except sqlite3.Error as e:
            logger.warning(f"query_products failed: {e}")
            return []


# Back-compat name used by ``agent6`` / docs
MemoryManager = MemoryService

# Legacy export: artifact text directory (binary store lives beside it)
ARTIFACTS_DIR = STATE_DIR / "artifacts"
