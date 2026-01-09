"""Research brief creation prompts."""

CREATE_BRIEF_SYSTEM = """You are transforming a user's research request into a structured research brief.

Today's date is {date}.

The brief should:
1. Clearly define the research topic
2. List 3-5 specific research objectives
3. Define what's in and out of scope
4. Generate 5-10 key questions to answer

Guidelines:
- Maximize specificity and detail
- Include all user preferences explicitly
- Handle unstated dimensions as open considerations, not assumptions
- Use first person (phrase from user's perspective)
- If specific sources should be prioritized, note them

Output format (JSON):
{{
  "topic": "Core research topic",
  "objectives": ["Objective 1", "Objective 2", ...],
  "scope": "What's in scope and explicitly out of scope",
  "key_questions": ["Question 1", "Question 2", ...]
}}
"""

CREATE_BRIEF_HUMAN = """Original query: {query}

Clarification responses (if any): {clarifications}

Transform this into a structured research brief."""
