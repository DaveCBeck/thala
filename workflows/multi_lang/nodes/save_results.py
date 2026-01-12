"""
Save multi-language research results to store.

Saves three types of records:
1. Per-language results (one per language)
2. Comparative analysis document
3. Final synthesized document
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from workflows.multi_lang.state import MultiLangState

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


async def save_multi_lang_results(state: MultiLangState) -> dict[str, Any]:
    """Save all results to the store.

    Saves THREE types of records:

    1. PER-LANGUAGE RECORDS (one per language in language_results)
       - id: new UUID
       - source_type: SourceType.INTERNAL
       - content: findings_summary
       - compression_level: 0
       - metadata: type, topic, language info, insights

    2. COMPARATIVE DOCUMENT RECORD
       - id: new UUID
       - source_type: SourceType.INTERNAL
       - content: sonnet_analysis.comparative_document
       - source_ids: all per-language record IDs (lineage)
       - metadata: type, topic, languages, themes, gaps

    3. SYNTHESIZED DOCUMENT RECORD
       - id: new UUID
       - source_type: SourceType.INTERNAL
       - content: final_synthesis
       - source_ids: all per-language record IDs (lineage)
       - metadata: type, topic, languages, workflows, integration info
    """
    try:
        from core.stores.schema import StoreRecord, SourceType
        from langchain_tools.base import get_store_manager

        store_manager = get_store_manager()
    except ImportError as e:
        logger.warning(f"Store components not available: {e}")
        return {
            "per_language_record_ids": {},
            "comparative_record_id": None,
            "synthesis_record_id": None,
            "completed_at": _utc_now(),
            "current_phase": "completed",
            "current_status": f"Completed (store not available: {e})",
            "errors": [{"phase": "save_results", "error": f"Store not available: {e}"}],
        }

    per_language_record_ids: dict[str, str] = {}
    errors: list[dict] = []
    topic = state["input"]["topic"]
    mode = state["input"]["mode"]
    workflows = state["input"]["workflows"]

    # 1. Save per-language records
    language_results = state.get("language_results", [])

    for result in language_results:
        lang_code = result["language_code"]
        content = result.get("findings_summary")

        if not content:
            logger.warning(f"No findings_summary for {lang_code}, skipping")
            continue

        try:
            # Generate embedding
            embed_text = content[:8000]
            embedding = await store_manager.embedding.embed(embed_text)

            record_id = uuid4()
            record = StoreRecord(
                id=record_id,
                source_type=SourceType.INTERNAL,
                content=content,
                compression_level=0,
                language_code=lang_code,
                metadata={
                    "type": "multi_lang_language_result",
                    "topic": topic,
                    "language_name": result["language_name"],
                    "workflows_run": result["workflows_run"],
                    "source_count": result["source_count"],
                    "key_insights": result["key_insights"],
                    "unique_perspectives": result["unique_perspectives"],
                },
                embedding_model=store_manager.embedding.model,
            )

            # Attach embedding to record and save to ES store
            record.embedding = embedding
            await store_manager.es_stores.store.add(record)

            per_language_record_ids[lang_code] = str(record_id)
            logger.info(f"Saved per-language record for {lang_code}: {record_id}")

        except Exception as e:
            logger.error(f"Failed to save per-language record for {lang_code}: {e}")
            errors.append({"phase": f"save_language_{lang_code}", "error": str(e)})

    # 2. Save comparative document record
    comparative_record_id: str | None = None
    sonnet_analysis = state.get("sonnet_analysis")

    if sonnet_analysis and sonnet_analysis.get("comparative_document"):
        try:
            content = sonnet_analysis["comparative_document"]
            embed_text = content[:8000]
            embedding = await store_manager.embedding.embed(embed_text)

            record_id = uuid4()
            # Extract language codes from per-language records
            languages_analyzed = list(per_language_record_ids.keys())

            record = StoreRecord(
                id=record_id,
                source_type=SourceType.INTERNAL,
                content=content,
                compression_level=0,
                source_ids=[uuid4() for _ in per_language_record_ids.values()],  # Lineage to per-language records
                metadata={
                    "type": "multi_lang_comparative",
                    "topic": topic,
                    "languages_analyzed": languages_analyzed,
                    "mode": mode,
                    "universal_themes": sonnet_analysis.get("universal_themes", []),
                    "coverage_gaps_in_english": sonnet_analysis.get("coverage_gaps_in_english", []),
                },
                embedding_model=store_manager.embedding.model,
            )

            record.embedding = embedding
            await store_manager.es_stores.store.add(record)

            comparative_record_id = str(record_id)
            logger.info(f"Saved comparative document: {record_id}")

        except Exception as e:
            logger.error(f"Failed to save comparative document: {e}")
            errors.append({"phase": "save_comparative", "error": str(e)})
    else:
        logger.warning("No comparative document available, skipping")

    # 3. Save synthesized document record
    synthesis_record_id: str | None = None
    final_synthesis = state.get("final_synthesis")

    if final_synthesis:
        try:
            embed_text = final_synthesis[:8000]
            embedding = await store_manager.embedding.embed(embed_text)

            record_id = uuid4()
            languages_integrated = list(per_language_record_ids.keys())
            integration_steps = state.get("integration_steps", [])

            # Determine workflows used (which were enabled)
            workflows_used = [k for k, v in workflows.items() if v]

            record = StoreRecord(
                id=record_id,
                source_type=SourceType.INTERNAL,
                content=final_synthesis,
                compression_level=0,
                source_ids=[uuid4() for _ in per_language_record_ids.values()],  # Lineage to per-language records
                metadata={
                    "type": "multi_lang_synthesis",
                    "topic": topic,
                    "languages_integrated": languages_integrated,
                    "mode": mode,
                    "workflows_used": workflows_used,
                    "integration_steps_count": len(integration_steps),
                },
                embedding_model=store_manager.embedding.model,
            )

            record.embedding = embedding
            await store_manager.es_stores.store.add(record)

            synthesis_record_id = str(record_id)
            logger.info(f"Saved synthesis document: {record_id}")

        except Exception as e:
            logger.error(f"Failed to save synthesis document: {e}")
            errors.append({"phase": "save_synthesis", "error": str(e)})
    else:
        logger.warning("No final synthesis available, skipping")

    result: dict[str, Any] = {
        "per_language_record_ids": per_language_record_ids,
        "comparative_record_id": comparative_record_id,
        "synthesis_record_id": synthesis_record_id,
        "completed_at": _utc_now(),
        "current_phase": "completed",
        "current_status": "All results saved to store",
    }

    if errors:
        result["errors"] = errors

    logger.info(
        f"Saved {len(per_language_record_ids)} per-language records, "
        f"comparative: {comparative_record_id is not None}, "
        f"synthesis: {synthesis_record_id is not None}"
    )

    return result
