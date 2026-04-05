"""
ContextLattice: Query-time context optimization for agentic LLM systems.

Implements a semantic hierarchy with vector ranking to minimize token usage
while preserving solution quality.
"""

__version__ = "0.1.0"

from .core.hierarchy import HierarchyLevel
from .core.node import ContextNode
from .core.budget import BudgetCalculator, ContextBudget
from .retrieval.intent_classifier import QueryIntent, IntentClassifier

__all__ = [
    "HierarchyLevel",
    "ContextNode",
    "BudgetCalculator",
    "ContextBudget",
    "QueryIntent",
    "IntentClassifier",
]
