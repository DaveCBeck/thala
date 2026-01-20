"""Execute edits nodes for editing workflow."""

import copy
import logging
from typing import Any

from langgraph.types import Send

from workflows.enhance.editing.document_model import DocumentModel, Section, ContentBlock, DocumentTransaction
from workflows.enhance.editing.schemas import EditPlan
from workflows.enhance.editing.prompts import (
    GENERATE_INTRODUCTION_SYSTEM,
    GENERATE_INTRODUCTION_USER,
    GENERATE_CONCLUSION_SYSTEM,
    GENERATE_CONCLUSION_USER,
    GENERATE_SYNTHESIS_SYSTEM,
    GENERATE_SYNTHESIS_USER,
    GENERATE_TRANSITION_SYSTEM,
    GENERATE_TRANSITION_USER,
    CONSOLIDATE_CONTENT_SYSTEM,
    CONSOLIDATE_CONTENT_USER,
)
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


def _flatten_sections(sections: list[Section]) -> list[Section]:
    """Recursively flatten sections including all subsections."""
    result = []
    for section in sections:
        result.append(section)
        if section.subsections:
            result.extend(_flatten_sections(section.subsections))
    return result


def route_to_edit_workers(state: dict) -> list[Send] | str:
    """Route to appropriate edit workers based on edit plan.

    Returns:
        List of Send objects for parallel workers, or "assemble_edits" if no edits
    """
    edit_plan_data = state.get("edit_plan", {})
    if not edit_plan_data:
        return "assemble_edits"

    edit_plan = EditPlan.model_validate(edit_plan_data)

    if not edit_plan.edits:
        return "assemble_edits"

    # Use updated document model if available (from previous iterations)
    current_doc_model = state.get("updated_document_model", state["document_model"])

    sends = []

    # Generation edits can run in parallel
    for i, edit in enumerate(edit_plan.generation_edits):
        sends.append(
            Send(
                "execute_generation_edit",
                {
                    "edit": edit.model_dump(),
                    "edit_index": i,
                    "document_model": current_doc_model,
                    "topic": state["input"]["topic"],
                    "quality_settings": state.get("quality_settings", {}),
                },
            )
        )

    # Structure edits run in single worker (sequential)
    if edit_plan.structure_edits:
        sends.append(
            Send(
                "execute_structure_edits",
                {
                    "edits": [e.model_dump() for e in edit_plan.structure_edits],
                    "document_model": current_doc_model,
                },
            )
        )

    # Removal edits can run in parallel
    for i, edit in enumerate(edit_plan.removal_edits):
        sends.append(
            Send(
                "execute_removal_edit",
                {
                    "edit": edit.model_dump(),
                    "edit_index": i,
                    "document_model": current_doc_model,
                },
            )
        )

    if not sends:
        return "assemble_edits"

    return sends


async def execute_generation_edit_worker(state: dict) -> dict[str, Any]:
    """Execute a content generation edit.

    Handles: generate_introduction, generate_conclusion,
    generate_synthesis, generate_transition
    """
    edit_data = state["edit"]
    edit_type = edit_data["edit_type"]
    document_model = DocumentModel.from_dict(state["document_model"])
    topic = state["topic"]
    quality_settings = state.get("quality_settings", {})

    use_opus = quality_settings.get("use_opus_for_generation", False)
    tier = ModelTier.OPUS if use_opus else ModelTier.SONNET

    logger.info(f"Executing generation edit: {edit_type}")

    try:
        if edit_type == "generate_introduction":
            # Get context sections
            context_content = ""
            for sec_id in edit_data.get("context_section_ids", []):
                section = document_model.get_section(sec_id)
                if section:
                    context_content += document_model.get_section_content(sec_id) + "\n\n"

            llm = get_llm(tier=tier, max_tokens=2000)
            user_prompt = GENERATE_INTRODUCTION_USER.format(
                scope=edit_data["scope"],
                topic=topic,
                context_content=context_content[:8000],  # Limit context
                requirements=edit_data["introduction_requirements"],
                target_words=edit_data["target_word_count"],
            )

            response = await llm.ainvoke([
                {"role": "system", "content": GENERATE_INTRODUCTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ])
            generated = response.content.strip()

        elif edit_type == "generate_conclusion":
            context_content = ""
            for sec_id in edit_data.get("context_section_ids", []):
                section = document_model.get_section(sec_id)
                if section:
                    context_content += document_model.get_section_content(sec_id) + "\n\n"

            llm = get_llm(tier=tier, max_tokens=2000)
            user_prompt = GENERATE_CONCLUSION_USER.format(
                scope=edit_data["scope"],
                topic=topic,
                context_content=context_content[:8000],
                requirements=edit_data["conclusion_requirements"],
                target_words=edit_data["target_word_count"],
            )

            response = await llm.ainvoke([
                {"role": "system", "content": GENERATE_CONCLUSION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ])
            generated = response.content.strip()

        elif edit_type == "generate_synthesis":
            section = document_model.get_section(edit_data["target_section_id"])
            section_content = document_model.get_section_content(
                edit_data["target_section_id"]
            ) if section else ""

            llm = get_llm(tier=tier, max_tokens=2000)
            user_prompt = GENERATE_SYNTHESIS_USER.format(
                topic=topic,
                section_content=section_content[:6000],
                requirements=edit_data["synthesis_requirements"],
                target_words=edit_data["target_word_count"],
            )

            response = await llm.ainvoke([
                {"role": "system", "content": GENERATE_SYNTHESIS_SYSTEM},
                {"role": "user", "content": user_prompt},
            ])
            generated = response.content.strip()

        elif edit_type == "generate_transition":
            from_section = document_model.get_section(edit_data["from_section_id"])
            to_section = document_model.get_section(edit_data["to_section_id"])

            from_content = document_model.get_section_content(
                edit_data["from_section_id"], include_subsections=False
            )[-2000:] if from_section else ""

            to_content = document_model.get_section_content(
                edit_data["to_section_id"], include_subsections=False
            )[:2000] if to_section else ""

            llm = get_llm(tier=tier, max_tokens=500)
            user_prompt = GENERATE_TRANSITION_USER.format(
                from_content=from_content,
                to_content=to_content,
                transition_type=edit_data["transition_type"],
                target_words=edit_data["target_word_count"],
            )

            response = await llm.ainvoke([
                {"role": "system", "content": GENERATE_TRANSITION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ])
            generated = response.content.strip()

        else:
            logger.warning(f"Unknown generation edit type: {edit_type}")
            return {"completed_edits": []}

        logger.info(f"Generated {len(generated.split())} words for {edit_type}")

        return {
            "completed_edits": [
                {
                    "edit_type": edit_type,
                    "success": True,
                    "generated_content": generated,
                    "target_section_id": edit_data.get("target_section_id"),
                    "insert_after_section_id": edit_data.get("insert_after_section_id"),
                    "new_section_heading": edit_data.get("new_section_heading"),
                    "context_section_ids": edit_data.get("context_section_ids", []),
                    "scope": edit_data.get("scope"),
                    "position": edit_data.get("position", "end"),
                    "word_count": len(generated.split()),
                }
            ]
        }

    except Exception as e:
        logger.error(f"Generation edit failed: {e}", exc_info=True)
        return {
            "completed_edits": [
                {
                    "edit_type": edit_type,
                    "success": False,
                    "error": str(e),
                }
            ],
            "errors": [{"node": "execute_generation_edit", "error": str(e)}],
        }


async def execute_structure_edits_worker(state: dict) -> dict[str, Any]:
    """Execute structural edits (moves, merges, consolidations) sequentially."""
    edits = state["edits"]
    document_model = DocumentModel.from_dict(state["document_model"])

    results = []

    for edit_data in edits:
        edit_type = edit_data["edit_type"]
        logger.info(f"Executing structure edit: {edit_type}")

        try:
            if edit_type == "section_move":
                # Section moves just record the operation
                # Actual move happens during assembly
                results.append({
                    "edit_type": edit_type,
                    "success": True,
                    "operation": "move",
                    "source_section_id": edit_data["source_section_id"],
                    "target_position": edit_data["target_position"],
                    "target_section_id": edit_data["target_section_id"],
                })

            elif edit_type == "section_merge":
                # Section merges are complex - we synthesize the content
                primary_section = document_model.get_section(edit_data["primary_section_id"])
                secondary_section = document_model.get_section(edit_data["secondary_section_id"])

                if primary_section and secondary_section:
                    primary_content = document_model.get_section_content(
                        edit_data["primary_section_id"]
                    )
                    secondary_content = document_model.get_section_content(
                        edit_data["secondary_section_id"]
                    )

                    # Use LLM to merge
                    llm = get_llm(tier=ModelTier.SONNET, max_tokens=4000)
                    response = await llm.ainvoke([
                        {"role": "system", "content": "Merge these two sections into one cohesive section. Eliminate redundancy while preserving all important information."},
                        {"role": "user", "content": f"PRIMARY SECTION:\n{primary_content}\n\nSECONDARY SECTION:\n{secondary_content}\n\nMerge strategy: {edit_data['merge_strategy']}\n\nCreate a single cohesive section."},
                    ])

                    results.append({
                        "edit_type": edit_type,
                        "success": True,
                        "operation": "merge",
                        "primary_section_id": edit_data["primary_section_id"],
                        "secondary_section_id": edit_data["secondary_section_id"],
                        "merged_content": response.content.strip(),
                        "new_heading": edit_data.get("new_heading"),
                    })
                else:
                    results.append({
                        "edit_type": edit_type,
                        "success": False,
                        "error": "Section not found",
                    })

            elif edit_type == "consolidate":
                # Consolidate scattered content
                source_contents = []
                for block_id in edit_data["source_block_ids"]:
                    block = document_model.get_block(block_id)
                    if block:
                        source_contents.append(block.content)

                if source_contents:
                    source_blocks_text = "\n\n---\n\n".join(source_contents)

                    llm = get_llm(tier=ModelTier.SONNET, max_tokens=3000)
                    user_prompt = CONSOLIDATE_CONTENT_USER.format(
                        topic=edit_data["topic"],
                        source_blocks=source_blocks_text,
                        approach=edit_data["consolidation_approach"],
                    )

                    response = await llm.ainvoke([
                        {"role": "system", "content": CONSOLIDATE_CONTENT_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ])

                    results.append({
                        "edit_type": edit_type,
                        "success": True,
                        "operation": "consolidate",
                        "source_block_ids": edit_data["source_block_ids"],
                        "target_section_id": edit_data["target_section_id"],
                        "consolidated_content": response.content.strip(),
                    })
                else:
                    results.append({
                        "edit_type": edit_type,
                        "success": False,
                        "error": "No source blocks found",
                    })

        except Exception as e:
            logger.error(f"Structure edit failed: {e}", exc_info=True)
            results.append({
                "edit_type": edit_type,
                "success": False,
                "error": str(e),
            })

    return {"completed_edits": results}


async def execute_removal_edit_worker(state: dict) -> dict[str, Any]:
    """Execute removal edits (delete, trim)."""
    edit_data = state["edit"]
    edit_type = edit_data["edit_type"]
    document_model = DocumentModel.from_dict(state["document_model"])

    logger.info(f"Executing removal edit: {edit_type}")

    try:
        if edit_type == "delete_redundant":
            # Validate blocks exist
            blocks_to_delete = []
            for block_id in edit_data["block_ids_to_delete"]:
                if document_model.get_block(block_id):
                    blocks_to_delete.append(block_id)

            return {
                "completed_edits": [
                    {
                        "edit_type": edit_type,
                        "success": True,
                        "operation": "delete",
                        "block_ids_deleted": blocks_to_delete,
                        "primary_block_id": edit_data["primary_block_id"],
                    }
                ]
            }

        elif edit_type == "trim_redundancy":
            block = document_model.get_block(edit_data["block_id"])
            if block:
                # Remove the specified content
                trimmed = block.content.replace(edit_data["content_to_remove"], "").strip()
                return {
                    "completed_edits": [
                        {
                            "edit_type": edit_type,
                            "success": True,
                            "operation": "trim",
                            "block_id": edit_data["block_id"],
                            "trimmed_content": trimmed,
                        }
                    ]
                }

        return {"completed_edits": [{"edit_type": edit_type, "success": False, "error": "Unknown removal type"}]}

    except Exception as e:
        logger.error(f"Removal edit failed: {e}", exc_info=True)
        return {
            "completed_edits": [{"edit_type": edit_type, "success": False, "error": str(e)}],
            "errors": [{"node": "execute_removal_edit", "error": str(e)}],
        }


def _find_last_content_section_id(document_model: DocumentModel) -> str | None:
    """Find the last content section (before References/Bibliography).

    This helper ensures generated content like conclusions don't end up
    in the References section.
    """
    all_sections = document_model.get_all_sections()
    if not all_sections:
        return None

    # Scan backwards to find last non-reference section
    for i in range(len(all_sections) - 1, -1, -1):
        heading_lower = all_sections[i].heading.lower()
        if not any(kw in heading_lower for kw in ["reference", "bibliography", "works cited"]):
            return all_sections[i].section_id

    # Fallback to last section if all are reference-like (unlikely)
    return all_sections[-1].section_id


def _find_section_in_list(sections: list[Section], section_id: str) -> Section | None:
    """Recursively find a section by ID in a list of sections."""
    for section in sections:
        if section.section_id == section_id:
            return section
        # Check subsections
        found = _find_section_in_list(section.subsections, section_id)
        if found:
            return found
    return None


async def assemble_edits_node(state: dict) -> dict[str, Any]:
    """Assemble completed edits into updated document model.

    This node takes all completed edits and applies them to create
    an updated document model using a transaction-like approach.
    """
    # Use deep copy to avoid mutating original state
    source_model_data = state.get("updated_document_model", state["document_model"])
    document_model = DocumentModel.from_dict(copy.deepcopy(source_model_data))
    completed_edits = state.get("completed_edits", [])

    if not completed_edits:
        logger.info("No edits to assemble")
        return {
            "updated_document_model": state["document_model"],
            "execution_complete": True,
        }

    successful_edits = [e for e in completed_edits if e.get("success")]
    logger.info(f"Assembling {len(successful_edits)} successful edits")

    # Track placement issues for verification
    placement_issues = []

    # Deep copy sections and preamble to avoid mutation issues
    new_sections = copy.deepcopy(document_model.sections)
    new_preamble = copy.deepcopy(document_model.preamble_blocks)

    for edit in successful_edits:
        edit_type = edit.get("edit_type", "")

        if edit_type == "generate_introduction":
            # Add introduction to preamble or section
            content = edit.get("generated_content", "")
            if content:
                if edit.get("scope") == "document":
                    # Add to beginning of preamble
                    new_block = ContentBlock.from_content(content, "paragraph")
                    new_preamble.insert(0, new_block)
                    logger.debug("Added document introduction to preamble")

        elif edit_type == "generate_conclusion":
            content = edit.get("generated_content", "")
            if not content:
                continue

            insert_after_id = edit.get("insert_after_section_id")
            target_section_id = edit.get("target_section_id")
            scope = edit.get("scope", "document")

            if insert_after_id:
                # Document scope: create a new conclusion section after the specified section
                new_heading = edit.get("new_section_heading", "Conclusion")
                new_section = Section.from_heading(new_heading, level=1)
                new_block = ContentBlock.from_content(content, "paragraph")
                new_section.blocks.append(new_block)

                # Find where to insert in the top-level sections list
                insert_idx = None
                for i, sec in enumerate(new_sections):
                    if sec.section_id == insert_after_id:
                        insert_idx = i + 1
                        break
                    # Also check subsections (flatten search)
                    all_in_sec = [sec] + list(_flatten_sections([sec]))
                    for sub in all_in_sec:
                        if sub.section_id == insert_after_id:
                            # Insert after this top-level section
                            insert_idx = i + 1
                            break
                    if insert_idx is not None:
                        break

                if insert_idx is not None:
                    new_sections.insert(insert_idx, new_section)
                    logger.debug(f"Created new conclusion section '{new_heading}' after section index {insert_idx - 1}")
                else:
                    # Smart fallback: find last content section (not References)
                    last_content_id = _find_last_content_section_id(document_model)
                    fallback_idx = len(new_sections)  # Default to end
                    if last_content_id:
                        for i, sec in enumerate(new_sections):
                            if sec.section_id == last_content_id:
                                fallback_idx = i + 1
                                break
                    new_sections.insert(fallback_idx, new_section)
                    logger.warning(f"Could not find insert_after_id {insert_after_id}, inserted conclusion at index {fallback_idx}")

            elif target_section_id:
                # Section scope: add to existing section
                # Find section in new_sections (deep copied)
                target_section = _find_section_in_list(new_sections, target_section_id)
                if target_section:
                    new_block = ContentBlock.from_content(content, "paragraph")
                    target_section.blocks.append(new_block)
                    logger.debug(f"Added conclusion to section '{target_section.heading}'")
                else:
                    placement_issues.append(f"Could not find target_section_id {target_section_id} for conclusion")
                    logger.warning(f"Could not find target_section_id {target_section_id}")

            elif scope == "document":
                # Document scope but no insert_after_id - use smart fallback
                new_heading = edit.get("new_section_heading", "Conclusion")
                new_section = Section.from_heading(new_heading, level=1)
                new_block = ContentBlock.from_content(content, "paragraph")
                new_section.blocks.append(new_block)

                # Find last content section (not References/Bibliography)
                last_content_id = _find_last_content_section_id(document_model)
                insert_idx = len(new_sections)  # Default to end
                if last_content_id:
                    for i, sec in enumerate(new_sections):
                        if sec.section_id == last_content_id:
                            insert_idx = i + 1
                            break
                        # Check subsections
                        all_in_sec = _flatten_sections([sec])
                        for sub in all_in_sec:
                            if sub.section_id == last_content_id:
                                insert_idx = i + 1
                                break
                        if insert_idx != len(new_sections):
                            break

                new_sections.insert(insert_idx, new_section)
                logger.debug(f"Created conclusion section '{new_heading}' at index {insert_idx} (smart fallback)")

            else:
                # Legacy fallback for section scope without target
                placement_issues.append("generate_conclusion missing both target_section_id and insert_after_section_id")
                logger.error("generate_conclusion edit missing both target_section_id and insert_after_section_id")

        elif edit_type == "generate_synthesis":
            content = edit.get("generated_content", "")
            target_id = edit.get("target_section_id")
            if content and target_id:
                # Use deep-copied section list
                section = _find_section_in_list(new_sections, target_id)
                if section:
                    new_block = ContentBlock.from_content(content, "paragraph")
                    if edit.get("position") == "start":
                        section.blocks.insert(0, new_block)
                    else:
                        section.blocks.append(new_block)
                    logger.debug(f"Added synthesis to section {target_id}")
                else:
                    placement_issues.append(f"Could not find target_section_id {target_id} for synthesis")
                    logger.warning(f"Could not find target section {target_id} for synthesis")

        elif edit_type == "generate_transition":
            content = edit.get("generated_content", "")
            from_id = edit.get("from_section_id")
            to_id = edit.get("to_section_id")
            if content and from_id:
                # Use deep-copied section list
                section = _find_section_in_list(new_sections, from_id)
                if section:
                    new_block = ContentBlock.from_content(content, "paragraph")
                    section.blocks.append(new_block)
                    logger.debug(f"Added transition after section {from_id}")
                else:
                    placement_issues.append(f"Could not find from_section_id {from_id} for transition")
                    logger.warning(f"Could not find from_section {from_id} for transition")

        # Note: section_move, section_merge, consolidate, delete would need
        # more complex handling - for now we log them

    # Create updated model from deep-copied sections
    updated_model = DocumentModel(
        title=document_model.title,
        sections=new_sections,
        preamble_blocks=new_preamble,
    )

    # Verify placement - check that conclusions didn't end up in References
    verification_result = _verify_edit_placements(updated_model, successful_edits)
    if verification_result["issues"]:
        placement_issues.extend(verification_result["issues"])

    logger.info(
        f"Assembled document: {updated_model.total_words} words, "
        f"{updated_model.section_count} sections"
    )
    if placement_issues:
        logger.warning(f"Placement issues detected: {placement_issues}")

    return {
        "updated_document_model": updated_model.to_dict(),
        "execution_complete": True,
        "placement_issues": placement_issues,
    }


def _verify_edit_placements(document_model: DocumentModel, edits: list[dict]) -> dict:
    """Verify that generated content was placed correctly.

    Returns dict with 'issues' list containing any detected problems.
    """
    issues = []

    # Find reference sections
    reference_sections = []
    for section in document_model.get_all_sections():
        heading_lower = section.heading.lower()
        if any(kw in heading_lower for kw in ["reference", "bibliography", "works cited"]):
            reference_sections.append(section)

    # Check each generation edit
    for edit in edits:
        edit_type = edit.get("edit_type", "")
        content = edit.get("generated_content", "")

        if not content or edit_type not in ("generate_conclusion", "generate_synthesis", "generate_introduction"):
            continue

        # Check first 100 chars of content to identify
        content_preview = content[:100]

        # Verify it's not in reference sections
        for ref_section in reference_sections:
            for block in ref_section.blocks:
                if content_preview in block.content:
                    issues.append(
                        f"{edit_type} content incorrectly placed in "
                        f"'{ref_section.heading}' section (ID: {ref_section.section_id})"
                    )
                    logger.error(
                        f"PLACEMENT ERROR: {edit_type} content found in "
                        f"'{ref_section.heading}' instead of proper location"
                    )

    return {"issues": issues}
