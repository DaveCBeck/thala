#!/usr/bin/env python3
"""Extract all flagged items from Loop 5 (Fact and Reference Checking) in a trace."""

import json
from pathlib import Path
from typing import Any
from collections import defaultdict


def main():
    trace_path = Path("/home/dave/thala/testing/traces/019ba18e-7124-7923-9dca-4565ead80738.json")
    output_path = Path("/home/dave/thala/testing/traces/loop5_flagged_items.json")

    print(f"Reading trace file: {trace_path}")
    with open(trace_path) as f:
        trace_data = json.load(f)

    trace_id = trace_data.get("run_id", "unknown")
    print(f"Trace ID: {trace_id}")

    # Get loop5_result from outputs
    outputs = trace_data.get("outputs", {})
    if not outputs or "loop5_result" not in outputs:
        print("ERROR: No loop5_result in outputs")
        return

    loop5_result = outputs["loop5_result"]
    print("Found loop5_result")

    # Extract data from loop5_result
    ambiguous_claims_raw = loop5_result.get("ambiguous_claims", [])
    todos_raw = loop5_result.get("todo_items", [])
    human_review_raw = loop5_result.get("human_review_items", [])

    print(f"\nRaw counts from loop5_result:")
    print(f"  Ambiguous claims: {len(ambiguous_claims_raw)}")
    print(f"  TODO items: {len(todos_raw)}")
    print(f"  Human review items: {len(human_review_raw)}")

    # Process ambiguous claims
    ambiguous_claims = []
    for i, claim_text in enumerate(ambiguous_claims_raw):
        if claim_text and isinstance(claim_text, str):
            # Parse claim text to extract details
            claim_entry = {
                "claim": claim_text,
                "index": i,
            }

            # Try to extract section or context from claim text
            if " - " in claim_text:
                parts = claim_text.split(" - ", 1)
                if len(parts) == 2:
                    claim_entry["claim"] = parts[0].strip()
                    claim_entry["reason"] = parts[1].strip()

            ambiguous_claims.append(claim_entry)

    # Process TODOs
    todos = []
    for i, todo_text in enumerate(todos_raw):
        if todo_text and isinstance(todo_text, str):
            todos.append({
                "todo_text": todo_text,
                "index": i,
            })

    # Process human review items
    human_review = []
    for i, item_text in enumerate(human_review_raw):
        if item_text and isinstance(item_text, str):
            review_entry = {
                "item": item_text,
                "index": i,
                "type": categorize_review_item(item_text),
            }
            human_review.append(review_entry)

    print(f"\nProcessed counts:")
    print(f"  Ambiguous claims: {len(ambiguous_claims)}")
    print(f"  TODO items: {len(todos)}")
    print(f"  Human review items: {len(human_review)}")

    # Categorize human review items by type
    review_by_type = defaultdict(int)
    for review in human_review:
        review_by_type[review["type"]] += 1

    # Analyze patterns
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    print("\nReview items by type:")
    for review_type, count in sorted(review_by_type.items(), key=lambda x: -x[1]):
        print(f"  {review_type}: {count}")

    # Sample some ambiguous claims
    print("\nSample ambiguous claims (first 5):")
    for i, claim in enumerate(ambiguous_claims[:5], 1):
        claim_text = claim["claim"]
        if len(claim_text) > 150:
            claim_text = claim_text[:150] + "..."
        print(f"  {i}. {claim_text}")
        if "reason" in claim:
            reason_text = claim["reason"]
            if len(reason_text) > 150:
                reason_text = reason_text[:150] + "..."
            print(f"     Reason: {reason_text}")

    # Sample some TODOs
    print("\nSample TODOs (first 5):")
    for i, todo in enumerate(todos[:5], 1):
        todo_text = todo["todo_text"]
        if len(todo_text) > 150:
            todo_text = todo_text[:150] + "..."
        print(f"  {i}. {todo_text}")

    # Sample human review items by type
    print("\nSample human review items by type:")
    for review_type in sorted(review_by_type.keys()):
        items_of_type = [r for r in human_review if r["type"] == review_type]
        print(f"\n  {review_type} ({len(items_of_type)} total):")
        for i, review in enumerate(items_of_type[:3], 1):
            item_text = review["item"]
            if len(item_text) > 120:
                item_text = item_text[:120] + "..."
            print(f"    {i}. {item_text}")

    # Build output structure
    output = {
        "trace_id": trace_id,
        "loop5_run_id": "019ba238-1151-70e1-be6a-f1cee85095f1",
        "ambiguous_claims": ambiguous_claims,
        "unaddressed_todos": todos,
        "human_review_items": human_review,
        "summary": {
            "total_ambiguous": len(ambiguous_claims),
            "total_todos": len(todos),
            "total_human_review": len(human_review),
            "review_by_type": dict(review_by_type),
        },
        "analysis": {
            "notes": generate_analysis_notes(ambiguous_claims, todos, human_review, review_by_type)
        }
    }

    print(f"\nWriting output to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput saved to: {output_path}")


def categorize_review_item(item: str) -> str:
    """Categorize a human review item by its content."""
    item_lower = item.lower()

    if "invalid edit" in item_lower:
        return "invalid_edit"
    elif "ambiguous claim" in item_lower:
        return "ambiguous_claim"
    elif "todo" in item_lower or "unaddressed" in item_lower:
        return "unaddressed_todo"
    elif "citation" in item_lower or "reference" in item_lower:
        return "citation_issue"
    else:
        return "other"


def generate_analysis_notes(claims: list, todos: list, reviews: list, review_by_type: dict) -> list[str]:
    """Generate analysis notes about the flagged items."""
    notes = []

    # Count citation-related issues
    citation_mentions = sum(1 for c in claims if "citation" in c["claim"].lower() or "[@" in c["claim"])
    notes.append(f"Citation-related ambiguous claims: {citation_mentions}/{len(claims)} ({100*citation_mentions//len(claims) if claims else 0}%)")

    # Check if claims about missing citations dominate
    missing_citation_claims = sum(1 for c in claims if "cannot be verified" in c["claim"].lower() or "could not be found" in c["claim"].lower() or "not available" in c["claim"].lower())
    notes.append(f"Claims about missing/unverifiable citations: {missing_citation_claims}/{len(claims)} ({100*missing_citation_claims//len(claims) if claims else 0}%)")

    # Check for claims about missing sources
    no_source_claims = sum(1 for c in claims if "no source" in c["claim"].lower() or "no citation" in c["claim"].lower() or "requires citation" in c["claim"].lower())
    notes.append(f"Claims flagged for missing sources: {no_source_claims}/{len(claims)} ({100*no_source_claims//len(claims) if claims else 0}%)")

    # Review item distribution
    top_review_type = max(review_by_type.items(), key=lambda x: x[1]) if review_by_type else ("none", 0)
    notes.append(f"Most common review item type: {top_review_type[0]} ({top_review_type[1]} items)")

    # Check for overly aggressive flagging patterns
    if claims and missing_citation_claims / len(claims) > 0.7:
        notes.append("PATTERN: High proportion of flags related to missing/unverifiable citations - may indicate corpus completeness issues rather than actual document problems")

    if reviews:
        ambiguous_in_reviews = sum(1 for r in reviews if r["type"] == "ambiguous_claim")
        if ambiguous_in_reviews / len(reviews) > 0.5:
            notes.append("PATTERN: Over 50% of human review items are ambiguous claims - may indicate overly aggressive ambiguity detection")

    return notes


if __name__ == "__main__":
    main()
