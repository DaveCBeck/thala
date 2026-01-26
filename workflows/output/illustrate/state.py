"""State schema for illustrate workflow."""

from operator import add
from typing import Annotated, Literal

from typing_extensions import TypedDict

from .config import IllustrateConfig
from .schemas import ImageLocationPlan


def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer that merges dictionaries from parallel nodes."""
    result = dict(left) if left else {}
    if right:
        result.update(right)
    return result


class IllustrateInput(TypedDict):
    """Input for the workflow."""

    markdown_document: str  # Raw markdown content
    title: str | None  # Document title (optional, extracted if not provided)
    output_dir: str | None  # Where to save images (optional)


class ImageGenResult(TypedDict):
    """Result from image generation attempt."""

    location_id: str
    success: bool
    image_bytes: bytes | None
    image_type: Literal["generated", "public_domain", "diagram"]
    prompt_or_query_used: str  # What was actually used to generate/find
    alt_text: str | None
    attribution: dict | None  # For public domain images


class ImageReviewResult(TypedDict):
    """Result from vision review."""

    location_id: str
    passed: bool
    severity: Literal["minor", "substantive"] | None  # If not passed
    issues: list[str]  # Identified problems
    improved_brief: str | None  # For retry


class FinalImage(TypedDict):
    """Final approved image for insertion."""

    location_id: str
    insertion_after_header: str
    file_path: str
    alt_text: str
    image_type: Literal["generated", "public_domain", "diagram"]
    attribution: dict | None


class WorkflowError(TypedDict):
    """Error or warning captured during workflow."""

    location_id: str | None
    severity: Literal["warning", "error"]
    message: str
    stage: str  # Which stage: "analysis", "generation", "review", "finalize"


class IllustrateState(TypedDict, total=False):
    """Main workflow state for document illustration.

    Uses Annotated[list[...], add] for parallel aggregation of results
    from nodes invoked via Send().
    """

    # Input
    input: IllustrateInput
    config: IllustrateConfig

    # Analysis phase
    extracted_title: str
    image_plan: list[ImageLocationPlan]  # All planned image locations

    # Generation phase (parallel aggregation via add reducer)
    generation_results: Annotated[list[ImageGenResult], add]

    # Review phase (parallel aggregation via add reducer)
    review_results: Annotated[list[ImageReviewResult], add]

    # Retry tracking (use reducers for parallel aggregation)
    retry_count: Annotated[dict[str, int], merge_dicts]  # location_id -> attempt count
    pending_retries: Annotated[list[str], add]  # location_ids needing retry
    retry_briefs: Annotated[dict[str, str], merge_dicts]  # location_id -> improved brief

    # Final output
    final_images: list[FinalImage]
    illustrated_document: str  # Markdown with image references inserted

    # Workflow metadata
    errors: Annotated[list[WorkflowError], add]
    status: Literal["success", "partial", "failed"]
