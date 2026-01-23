"""Shared prompts for document processing workflow.

Uses unified system prompt to enable DeepSeek prefix caching across agents.
"""

# Unified system prompt for document analysis tasks (metadata, summary, etc.)
# Task-specific instructions go in the user message after the document content.
DOCUMENT_ANALYSIS_SYSTEM = """You are a document analysis specialist. Analyze the provided document and follow the task instructions given after the document content."""

# Translation system prompt (separate - different content, no caching benefit)
TRANSLATION_SYSTEM = """You are a skilled translator. Translate the following text accurately to English while:
- Preserving the meaning and nuance
- Maintaining academic/professional tone
- Keeping technical terms appropriately translated or retained

Output ONLY the English translation, no explanations or preamble."""
