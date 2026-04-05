"""
ContextNode: A single piece of retrievable context with metadata.

Nodes are organized into a semantic hierarchy and ranked within their level
using vector similarity and within-level weights.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import math

from .hierarchy import HierarchyLevel


@dataclass
class ContextNode:
    """
    A single piece of context that can be included in a query.

    Each node has:
    - Content (the actual text)
    - Hierarchy level (structural importance)
    - Vector representation (for similarity)
    - Within-level weights (recency, usage, user corrections)
    """

    # Identity
    id: str
    content: str
    tokens: int

    # Hierarchy membership
    level: HierarchyLevel

    # Vector representation (384-dim from all-MiniLM-L6-v2)
    embedding: Optional[np.ndarray] = None

    # Within-level weights (same-type factors only)
    recency_score: float = 1.0       # 0-1, exponential decay based on age
    usage_count: int = 0             # Times referenced in responses
    user_boost: float = 1.0          # 1.0 default, 1.5 if user-corrected

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Validate and set defaults."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

        # Calculate recency score if not set
        if self.recency_score == 1.0 and self.timestamp:
            self.recency_score = self._calculate_recency()

    def _calculate_recency(self, half_life_days: int = 30) -> float:
        """
        Calculate exponential decay based on age.

        Args:
            half_life_days: Number of days for score to decay to 0.5

        Returns:
            Recency score in range (0, 1]
        """
        if not self.timestamp:
            return 1.0

        age_days = (datetime.now() - self.timestamp).days
        if age_days < 0:
            age_days = 0

        # Exponential decay: score = exp(-age / half_life)
        decay_rate = math.log(2) / half_life_days
        score = math.exp(-age_days * decay_rate)

        return max(0.01, min(1.0, score))  # Clamp to (0.01, 1.0]

    @property
    def within_level_weight(self) -> float:
        """
        Composite weight for ranking within same hierarchy level.

        Combines:
        - Recency: Newer is better
        - Usage: Frequently referenced is better
        - User boost: User-corrected content is better

        Returns:
            Combined weight (0, ~3.0]
        """
        # Log(1 + usage) to avoid over-weighting high-usage nodes
        usage_factor = math.log(1 + self.usage_count)

        return self.recency_score * self.user_boost * (1 + usage_factor)

    def increment_usage(self):
        """Increment usage count (called when this node is referenced)."""
        self.usage_count += 1

    def apply_user_boost(self, factor: float = 1.5):
        """Apply user feedback boost (called when user says this was helpful)."""
        self.user_boost = factor

    def get_similarity(self, query_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity to query.

        Args:
            query_embedding: Query embedding vector (same dim as self.embedding)

        Returns:
            Cosine similarity in [-1, 1]
        """
        if self.embedding is None:
            raise ValueError(f"Node {self.id} has no embedding")

        if query_embedding is None:
            raise ValueError("Query embedding is None")

        # Cosine similarity
        dot_product = np.dot(self.embedding, query_embedding)
        norm_product = np.linalg.norm(self.embedding) * np.linalg.norm(query_embedding)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for storage/logging)."""
        return {
            "id": self.id,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "tokens": self.tokens,
            "level": self.level.name,
            "recency_score": round(self.recency_score, 3),
            "usage_count": self.usage_count,
            "user_boost": self.user_boost,
            "within_level_weight": round(self.within_level_weight, 3),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ContextNode(id={self.id}, level={self.level.name}, "
            f"tokens={self.tokens}, weight={self.within_level_weight:.3f})"
        )
