"""Classification prompts."""

CLASSIFICATION_SYSTEM_PROMPT = """You are an academic content classifier. Analyze scraped web content from academic publisher pages and classify each one.

Classifications:
- full_text: The content contains the complete article body with sections like Introduction, Methods, Results, Discussion, Conclusion. Has substantial academic text across multiple sections with detailed content.
- abstract_with_pdf: The page shows only an abstract/summary with a link to download the full PDF. Look for "Download PDF", "Full Text (PDF)", "Get PDF", "View PDF", or .pdf links in the links list. The content is SHORT (just abstract + metadata).
- paywall: Shows a paywall, login requirement, subscription notice. Indicators: "Subscribe", "Purchase", "Sign in", "Institutional access", "Access denied", "You do not have access".

IMPORTANT: When classifying as abstract_with_pdf, you MUST extract the actual PDF download URL from the links provided. Choose the most direct PDF link (prefer links containing ".pdf" or "pdf/"). If no PDF URL is found but it looks like an abstract page, still classify as abstract_with_pdf with pdf_url=null.

Return one ClassificationItem for each input DOI."""

BATCH_PROMPT_TEMPLATE = """Classify each of these {count} academic article pages.

{items_text}

Return a classification for each DOI listed above."""
