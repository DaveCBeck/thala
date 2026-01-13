"""Final report generation and refinement prompts."""

FINAL_REPORT_SYSTEM_STATIC = """Generate a comprehensive research report with proper citations.

Today's date is {date}.

<Requirements>
1. Start with an executive summary
2. Organize findings logically by theme or question
3. **Integrate memory context** - note what confirms/contradicts existing knowledge
4. Use inline citations [1], [2] etc.
5. End with a References section with full URLs
6. Note remaining uncertainties or areas for future research

Be INSIGHTFUL, not just comprehensive. Highlight:
- Key insights the user should know
- Surprising findings
- Connections to their existing knowledge
- Practical implications
</Requirements>

<Insightfulness Rules>
- Granular breakdown of topics with specific causes and impacts
- Detailed mapping tables where appropriate
- Nuanced discussion with explicit exploration of complexities
</Insightfulness Rules>

<Helpfulness Rules>
- Directly address the user's request
- Fluent, coherent, logically structured
- Accurate facts and reasoning
- Professional tone without unnecessary jargon
</Helpfulness Rules>

<Citation Rules>
- Assign each unique URL a single citation number
- End with ### Sources listing each source with numbers
- Number sources sequentially (1, 2, 3...) without gaps
- Format: [1] Source Title: URL
</Citation Rules>

<Structure Options>
For comparisons: intro → overview A → overview B → comparison → conclusion
For lists: just the list/table, or separate sections per item
For summaries: overview → concept 1 → concept 2 → ... → conclusion
For simple answers: single section

Choose the structure that best fits the content.
</Structure Options>

Write in clear paragraphs (not bullet lists by default).
Do NOT refer to yourself as the writer.
Each section should be thorough - users expect depth.
"""

FINAL_REPORT_USER_TEMPLATE = """Generate the final research report based on the following:

<Research Brief>
{research_brief}
</Research Brief>

<All Research Findings>
{all_findings}
</All Research Findings>

<Draft Report (refined through iterations)>
{draft_report}
</Draft Report>

<Memory Context (what the user already knew)>
{memory_context}
</Memory Context>

Make it genuinely useful and insightful for this specific user."""

FINAL_REPORT_SYSTEM = """Generate a comprehensive research report with proper citations.

Today's date is {date}.

<Research Brief>
{research_brief}
</Research Brief>

<All Research Findings>
{all_findings}
</All Research Findings>

<Draft Report (refined through iterations)>
{draft_report}
</Draft Report>

<Memory Context (what the user already knew)>
{memory_context}
</Memory Context>

<Requirements>
1. Start with an executive summary
2. Organize findings logically by theme or question
3. **Integrate memory context** - note what confirms/contradicts existing knowledge
4. Use inline citations [1], [2] etc.
5. End with a References section with full URLs
6. Note remaining uncertainties or areas for future research

Be INSIGHTFUL, not just comprehensive. Highlight:
- Key insights the user should know
- Surprising findings
- Connections to their existing knowledge
- Practical implications
</Requirements>

<Insightfulness Rules>
- Granular breakdown of topics with specific causes and impacts
- Detailed mapping tables where appropriate
- Nuanced discussion with explicit exploration of complexities
</Insightfulness Rules>

<Helpfulness Rules>
- Directly address the user's request
- Fluent, coherent, logically structured
- Accurate facts and reasoning
- Professional tone without unnecessary jargon
</Helpfulness Rules>

<Citation Rules>
- Assign each unique URL a single citation number
- End with ### Sources listing each source with numbers
- Number sources sequentially (1, 2, 3...) without gaps
- Format: [1] Source Title: URL
</Citation Rules>

<Structure Options>
For comparisons: intro → overview A → overview B → comparison → conclusion
For lists: just the list/table, or separate sections per item
For summaries: overview → concept 1 → concept 2 → ... → conclusion
For simple answers: single section

Choose the structure that best fits the content.
</Structure Options>

Write in clear paragraphs (not bullet lists by default).
Do NOT refer to yourself as the writer.
Each section should be thorough - users expect depth.
"""

FINAL_REPORT_HUMAN = """Generate the final research report now. Make it genuinely useful and insightful for this specific user."""

REFINE_DRAFT_SYSTEM = """Refine the draft report based on new research findings.

Today's date is {date}.

<Research Brief>
{research_brief}
</Research Brief>

<Current Draft>
{draft_report}
</Current Draft>

<New Findings>
{new_findings}
</New Findings>

<Task>
Update the draft report to incorporate the new findings:
1. Add new information in the appropriate sections
2. Resolve any contradictions with existing content
3. Update citations
4. Identify remaining gaps

Output the complete updated draft (not just the changes).
</Task>
"""
