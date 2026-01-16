"""Agent execution types."""

from pydantic import BaseModel


class AgentBudget(BaseModel):
    """Budget tracking for agent execution."""

    max_tool_calls: int = 12
    max_total_chars: int = 0  # 0 = unlimited (no character budget)
    tool_calls_made: int = 0
    chars_retrieved: int = 0

    def can_continue(self) -> bool:
        """Check if budget allows more tool calls.

        Only enforces tool call count. Character budget disabled (set to 0 for unlimited).
        """
        calls_ok = self.tool_calls_made < self.max_tool_calls
        # Character budget disabled - always returns True
        chars_ok = self.max_total_chars == 0 or self.chars_retrieved < self.max_total_chars
        return calls_ok and chars_ok

    def is_char_budget_exceeded(self) -> bool:
        """Check if character budget specifically was exceeded."""
        # With max_total_chars=0, this is always False (unlimited)
        return self.max_total_chars > 0 and self.chars_retrieved >= self.max_total_chars

    def get_remaining_char_budget(self) -> int:
        """Get remaining character budget."""
        return max(0, self.max_total_chars - self.chars_retrieved)

    def record_tool_call(self, result_chars: int) -> None:
        """Record a tool call and its result size."""
        self.tool_calls_made += 1
        self.chars_retrieved += result_chars

    def get_status(self) -> str:
        """Get budget status string."""
        return (
            f"[Budget: {self.tool_calls_made}/{self.max_tool_calls} calls, "
            f"{self.chars_retrieved}/{self.max_total_chars} chars]"
        )


__all__ = ["AgentBudget"]
