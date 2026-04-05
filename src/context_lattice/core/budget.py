"""
Budget calculation for context optimization.

Determines how many tokens are available for context after accounting for
conversation history, tool definitions, and system prompts.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .hierarchy import HierarchyLevel, HierarchyConfig


# Constants
MAX_TOKENS = 200_000  # Claude Opus 4.5 context window
MIN_VIABLE_TOKENS = 1_000  # Minimum for any useful context
RESERVED_FOR_RESPONSE = 4_000  # Reserve tokens for response generation


@dataclass
class ContextBudget:
    """
    Token budget allocation across hierarchy levels.

    Tracks:
    - Total available tokens
    - Allocation per hierarchy level
    - Remaining budget after allocation
    """

    total_available: int
    per_level: Dict[HierarchyLevel, int]
    reserved_for_response: int = RESERVED_FOR_RESPONSE

    @property
    def total_allocated(self) -> int:
        """Sum of all per-level allocations."""
        return sum(self.per_level.values())

    @property
    def is_minimal(self) -> bool:
        """Check if this is a minimal emergency budget."""
        return self.total_available < MIN_VIABLE_TOKENS * 2

    def to_dict(self) -> Dict[str, any]:
        """Serialize to dictionary for logging."""
        return {
            "total_available": self.total_available,
            "total_allocated": self.total_allocated,
            "reserved_for_response": self.reserved_for_response,
            "per_level": {
                level.name: tokens
                for level, tokens in self.per_level.items()
            },
            "is_minimal": self.is_minimal,
        }


class BudgetCalculator:
    """
    Calculate available token budget for context.

    Accounts for:
    - Conversation history
    - Tool definitions
    - System prompts
    - Reserved tokens for response generation
    """

    def __init__(self, config: Optional[HierarchyConfig] = None):
        """
        Initialize budget calculator.

        Args:
            config: Hierarchy configuration for budget allocation
        """
        self.config = config or HierarchyConfig()
        self.config.validate()

    def calculate(
        self,
        conversation_tokens: int = 0,
        tools_tokens: int = 0,
        system_tokens: int = 0,
        intent: str = "UNKNOWN",
    ) -> ContextBudget:
        """
        Calculate available budget for context.

        Args:
            conversation_tokens: Tokens used by conversation history
            tools_tokens: Tokens used by tool definitions
            system_tokens: Tokens used by system prompts
            intent: Query intent for intent-specific allocation

        Returns:
            ContextBudget with per-level allocations
        """
        # Calculate available tokens
        used_tokens = conversation_tokens + tools_tokens + system_tokens
        available = MAX_TOKENS - used_tokens - RESERVED_FOR_RESPONSE

        # Graceful degradation for low budgets
        if available < MIN_VIABLE_TOKENS:
            return self._minimal_budget()

        # Get per-level allocation based on intent
        per_level = self.config.get_budget_allocation(available, intent)

        return ContextBudget(
            total_available=available,
            per_level=per_level,
            reserved_for_response=RESERVED_FOR_RESPONSE,
        )

    def _minimal_budget(self) -> ContextBudget:
        """
        Create minimal emergency budget.

        When token budget is critically low, allocate only to STRUCTURAL
        level (user preferences, project context) to ensure basic functionality.

        Returns:
            Minimal ContextBudget with only STRUCTURAL allocation
        """
        return ContextBudget(
            total_available=MIN_VIABLE_TOKENS,
            per_level={
                HierarchyLevel.STRUCTURAL: MIN_VIABLE_TOKENS,
                HierarchyLevel.DIRECT: 0,
                HierarchyLevel.IMPLIED: 0,
                HierarchyLevel.BACKGROUND: 0,
            },
            reserved_for_response=RESERVED_FOR_RESPONSE,
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses simple heuristic: ~4 characters per token (conservative).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Simple heuristic: 4 chars per token
        # This is conservative (actual is ~3.5-4 for English)
        return len(text) // 4 + 1

    def fits_in_budget(
        self,
        tokens: int,
        level: HierarchyLevel,
        budget: ContextBudget,
    ) -> bool:
        """
        Check if content fits in level budget.

        Args:
            tokens: Token count to check
            level: Hierarchy level
            budget: Current budget allocation

        Returns:
            True if content fits, False otherwise
        """
        return tokens <= budget.per_level.get(level, 0)
