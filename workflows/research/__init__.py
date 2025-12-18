"""
Deep Research Workflow.

Implements a Self-Balancing Diffusion Algorithm for comprehensive research:
1. Clarify user intent
2. Create research brief
3. Search Thala memory stores for existing knowledge
4. Iterate/customize plan based on memory context
5. Supervisor coordinates parallel researcher agents
6. Generate final report with citations
7. Save findings to store
"""

from workflows.research.graph import deep_research, DeepResearchState

__all__ = ["deep_research", "DeepResearchState"]
