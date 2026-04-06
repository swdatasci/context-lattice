"""
Semantic hierarchy definitions for context organization.

The hierarchy defines 4 levels of context based on structural importance,
not just relevance scores. This ensures critical context (user preferences,
project constraints) is always included regardless of semantic similarity.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict


class HierarchyLevel(Enum):
    """
    Four-level semantic hierarchy for context organization.

    Each level has different inclusion criteria and budget allocation:

    - STRUCTURAL: Always included (user prefs, project CLAUDE.md, current task)
    - DIRECT: Explicitly mentioned in query (files, entities, recent turns)
    - IMPLIED: Semantically related (same module, related types, tests)
    - BACKGROUND: If space allows (architecture, conventions, history)
    """

    STRUCTURAL = 0  # Always included, fixed budget
    DIRECT = 1      # Query-matched, high priority
    IMPLIED = 2     # Semantically related, medium priority
    BACKGROUND = 3  # Historical/architectural, low priority

    @property
    def description(self) -> str:
        """Human-readable description of this level."""
        descriptions = {
            HierarchyLevel.STRUCTURAL: "Always included (user preferences, project context)",
            HierarchyLevel.DIRECT: "Query-matched (files, entities, recent conversation)",
            HierarchyLevel.IMPLIED: "Semantically related (same module, related types)",
            HierarchyLevel.BACKGROUND: "If space allows (architecture, conventions)",
        }
        return descriptions[self]

    @property
    def default_budget_pct(self) -> float:
        """Default budget allocation percentage for this level."""
        allocations = {
            HierarchyLevel.STRUCTURAL: 0.15,  # 15% - fixed, always included
            HierarchyLevel.DIRECT: 0.45,      # 45% - highest priority
            HierarchyLevel.IMPLIED: 0.30,     # 30% - medium priority
            HierarchyLevel.BACKGROUND: 0.10,  # 10% - lowest priority
        }
        return allocations[self]


@dataclass
class HierarchyConfig:
    """
    Configuration for hierarchy-based context selection.

    Defines budget allocation, thresholds, and pool weights for each level.
    """

    # Budget allocation per level (should sum to 1.0)
    structural_pct: float = 0.15
    direct_pct: float = 0.45
    implied_pct: float = 0.30
    background_pct: float = 0.10

    # Relevance thresholds for IMPLIED and BACKGROUND levels
    # STRUCTURAL and DIRECT don't use thresholds (rule-based inclusion)
    # Lowered from 0.6/0.4 to reduce over-aggressive filtering
    implied_threshold: float = 0.3   # Cosine similarity threshold
    background_threshold: float = 0.2

    # Intent-specific pool weights (multipliers applied to budget allocation)
    intent_weights: Dict[str, Dict[HierarchyLevel, float]] = None

    def __post_init__(self):
        """Initialize intent-specific weights if not provided."""
        if self.intent_weights is None:
            self.intent_weights = {
                "DEBUGGING": {
                    HierarchyLevel.STRUCTURAL: 1.0,
                    HierarchyLevel.DIRECT: 1.5,      # Boost: the buggy file
                    HierarchyLevel.IMPLIED: 1.2,     # Boost: related code
                    HierarchyLevel.BACKGROUND: 0.5,  # Reduce: docs less useful
                },
                "RESEARCH": {
                    HierarchyLevel.STRUCTURAL: 1.0,
                    HierarchyLevel.DIRECT: 0.8,
                    HierarchyLevel.IMPLIED: 1.0,
                    HierarchyLevel.BACKGROUND: 1.5,  # Boost: docs very useful
                },
                "CODING": {
                    HierarchyLevel.STRUCTURAL: 1.0,
                    HierarchyLevel.DIRECT: 1.5,      # Boost: need more budget for files
                    HierarchyLevel.IMPLIED: 1.2,     # Boost: related code important
                    HierarchyLevel.BACKGROUND: 0.7,  # Reduce: docs less critical
                },
                "REFACTORING": {
                    HierarchyLevel.STRUCTURAL: 1.0,
                    HierarchyLevel.DIRECT: 1.3,
                    HierarchyLevel.IMPLIED: 1.3,     # Boost: need related code
                    HierarchyLevel.BACKGROUND: 0.7,
                },
                "PLANNING": {
                    HierarchyLevel.STRUCTURAL: 1.2,  # Boost: project context critical
                    HierarchyLevel.DIRECT: 0.9,
                    HierarchyLevel.IMPLIED: 0.8,
                    HierarchyLevel.BACKGROUND: 1.4,  # Boost: architecture matters
                },
                "DOCUMENTATION": {
                    HierarchyLevel.STRUCTURAL: 1.0,
                    HierarchyLevel.DIRECT: 1.1,
                    HierarchyLevel.IMPLIED: 0.9,
                    HierarchyLevel.BACKGROUND: 1.2,  # Boost: conventions matter
                },
                "UNKNOWN": {
                    HierarchyLevel.STRUCTURAL: 1.0,
                    HierarchyLevel.DIRECT: 1.0,
                    HierarchyLevel.IMPLIED: 1.0,
                    HierarchyLevel.BACKGROUND: 1.0,
                },
            }

    def get_budget_allocation(self, total_tokens: int, intent: str = "UNKNOWN") -> Dict[HierarchyLevel, int]:
        """
        Calculate token budget for each hierarchy level.

        Args:
            total_tokens: Total available tokens for context
            intent: Query intent (e.g., "DEBUGGING", "RESEARCH")

        Returns:
            Dictionary mapping HierarchyLevel to token budget
        """
        weights = self.intent_weights.get(intent, self.intent_weights["UNKNOWN"])

        # Base allocation
        base_allocation = {
            HierarchyLevel.STRUCTURAL: self.structural_pct,
            HierarchyLevel.DIRECT: self.direct_pct,
            HierarchyLevel.IMPLIED: self.implied_pct,
            HierarchyLevel.BACKGROUND: self.background_pct,
        }

        # Apply intent-specific weights
        weighted_allocation = {
            level: pct * weights[level]
            for level, pct in base_allocation.items()
        }

        # Normalize to sum to 1.0
        total_weight = sum(weighted_allocation.values())
        normalized = {
            level: pct / total_weight
            for level, pct in weighted_allocation.items()
        }

        # Convert to token budgets
        return {
            level: int(total_tokens * pct)
            for level, pct in normalized.items()
        }

    def validate(self) -> bool:
        """Validate configuration parameters."""
        total_pct = self.structural_pct + self.direct_pct + self.implied_pct + self.background_pct

        if not (0.99 <= total_pct <= 1.01):  # Allow small floating point error
            raise ValueError(f"Budget percentages must sum to 1.0, got {total_pct}")

        if not (0.0 <= self.implied_threshold <= 1.0):
            raise ValueError(f"implied_threshold must be in [0, 1], got {self.implied_threshold}")

        if not (0.0 <= self.background_threshold <= 1.0):
            raise ValueError(f"background_threshold must be in [0, 1], got {self.background_threshold}")

        return True
