"""
Quality analysis utilities for test scripts.

Provides a base framework for analyzing workflow output quality,
identifying issues, and generating improvement suggestions.
"""

from dataclasses import dataclass, field


@dataclass
class QualityMetrics:
    """Container for quality analysis results.

    Attributes:
        completed: Whether the workflow completed successfully
        output_length: Length of primary output in characters
        word_count: Word count of primary output
        source_count: Number of sources/references
        error_count: Number of errors encountered
        issues: List of identified quality issues
        suggestions: List of improvement suggestions
        workflow_specific: Dict for workflow-specific metrics
    """
    completed: bool = False
    output_length: int = 0
    word_count: int = 0
    source_count: int = 0
    error_count: int = 0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    workflow_specific: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metrics": {
                "completed": self.completed,
                "output_length": self.output_length,
                "word_count": self.word_count,
                "source_count": self.source_count,
                "error_count": self.error_count,
                **self.workflow_specific,
            },
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


class BaseQualityAnalyzer:
    """Base class for workflow quality analysis.

    Provides common analysis logic with extension points for
    workflow-specific metrics. Subclass and override methods
    as needed for specific workflows.

    Usage:
        analyzer = BaseQualityAnalyzer(result)
        metrics = analyzer.analyze()
        print_quality_analysis(metrics)
    """

    # Field names for the primary output (override in subclasses)
    output_field: str = "final_report"
    output_field_alt: str | None = None  # Alternative field name

    # Thresholds for issue detection (override in subclasses)
    min_word_count: int = 500
    max_word_count: int = 10000
    min_source_count: int = 3

    def __init__(self, result: dict):
        """Initialize analyzer with workflow result.

        Args:
            result: Workflow result dictionary
        """
        self.result = result

    def analyze(self) -> QualityMetrics:
        """Run full quality analysis.

        Returns:
            QualityMetrics with analysis results
        """
        metrics = QualityMetrics()

        # Base analysis
        self._check_completion(metrics)
        self._analyze_output(metrics)
        self._count_sources(metrics)
        self._count_errors(metrics)

        # Workflow-specific analysis (override in subclasses)
        self._analyze_workflow_specific(metrics)

        # Generate issues and suggestions
        self._identify_issues(metrics)
        self._generate_suggestions(metrics)

        return metrics

    def _get_output(self) -> str:
        """Get the primary output field value.

        Returns:
            Output string or empty string
        """
        output = self.result.get(self.output_field, "")
        if not output and self.output_field_alt:
            output = self.result.get(self.output_field_alt, "")
        return output or ""

    def _check_completion(self, metrics: QualityMetrics) -> None:
        """Check if workflow completed successfully."""
        output = self._get_output()
        # Check for failure indicators
        if output and not output.startswith(("Failed", "Error", "# ") or "failed" in output[:100].lower()):
            metrics.completed = True

        status = self.result.get("current_status", "")
        if status == "completed":
            metrics.completed = True

    def _analyze_output(self, metrics: QualityMetrics) -> None:
        """Analyze the primary output."""
        output = self._get_output()
        if output:
            metrics.output_length = len(output)
            metrics.word_count = len(output.split())

    def _count_sources(self, metrics: QualityMetrics) -> None:
        """Count sources/references."""
        # Try common source fields
        citations = self.result.get("citations", [])
        references = self.result.get("references", [])
        sources = self.result.get("sources", [])

        metrics.source_count = max(len(citations), len(references), len(sources))

    def _count_errors(self, metrics: QualityMetrics) -> None:
        """Count errors encountered."""
        errors = self.result.get("errors", [])
        metrics.error_count = len(errors)

    def _analyze_workflow_specific(self, metrics: QualityMetrics) -> None:
        """Override in subclasses for workflow-specific analysis."""
        pass

    def _identify_issues(self, metrics: QualityMetrics) -> None:
        """Identify quality issues based on metrics."""
        if not metrics.completed:
            metrics.issues.append("Workflow did not complete successfully")

        if metrics.word_count == 0:
            metrics.issues.append("No output generated")
        elif metrics.word_count < self.min_word_count:
            metrics.issues.append(f"Output is short ({metrics.word_count} words)")
        elif metrics.word_count > self.max_word_count:
            metrics.issues.append(f"Output may be too long ({metrics.word_count} words)")

        if metrics.source_count < self.min_source_count:
            metrics.issues.append(f"Low source count ({metrics.source_count})")

        if metrics.error_count > 0:
            metrics.issues.append(f"{metrics.error_count} errors encountered")

    def _generate_suggestions(self, metrics: QualityMetrics) -> None:
        """Generate improvement suggestions based on issues."""
        if not metrics.issues:
            metrics.suggestions.append("Workflow completed successfully - no major issues detected")
            return

        if "short" in str(metrics.issues).lower():
            metrics.suggestions.append("Consider using a higher quality setting")

        if "source count" in str(metrics.issues).lower():
            metrics.suggestions.append("Consider broadening search terms or using more sources")

        if "errors" in str(metrics.issues).lower():
            metrics.suggestions.append("Review error logs for specific failures")


def print_quality_analysis(metrics: QualityMetrics | dict) -> None:
    """Print formatted quality analysis.

    Args:
        metrics: QualityMetrics instance or dict from to_dict()
    """
    print("\n" + "=" * 80)
    print("QUALITY ANALYSIS")
    print("=" * 80)

    # Handle both QualityMetrics and dict
    if isinstance(metrics, QualityMetrics):
        data = metrics.to_dict()
    else:
        data = metrics

    # Metrics
    print("\n--- Metrics ---")
    metrics_dict = data.get("metrics", {})
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Issues
    issues = data.get("issues", [])
    if issues:
        print(f"\n--- Issues Found ({len(issues)}) ---")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n--- No Issues Found ---")

    # Suggestions
    suggestions = data.get("suggestions", [])
    if suggestions:
        print(f"\n--- Suggestions ---")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

    print("\n" + "=" * 80)
