"""
Vector ranking: Rank context nodes within hierarchy pools.

Uses cached embeddings for efficiency (cost-aware optimization level 1).
Combines semantic similarity with within-level weights (recency, usage, user boost).
"""

from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

from ..core.hierarchy import HierarchyLevel, HierarchyConfig
from ..core.node import ContextNode


@dataclass
class RankedNode:
    """A context node with ranking score."""

    node: ContextNode
    similarity_score: float  # Cosine similarity to query
    weight_score: float      # Within-level weight
    final_score: float       # Combined score

    def __repr__(self) -> str:
        return (
            f"RankedNode(id={self.node.id}, "
            f"sim={self.similarity_score:.3f}, "
            f"weight={self.weight_score:.3f}, "
            f"final={self.final_score:.3f})"
        )


class VectorRanker:
    """
    Rank context nodes within hierarchy pools using vectors.

    Uses cached embeddings (optimization level 1) for efficiency.
    Combines semantic similarity with within-level weights.
    """

    def __init__(self, config: Optional[HierarchyConfig] = None):
        """
        Initialize vector ranker.

        Args:
            config: Hierarchy configuration
        """
        self.config = config or HierarchyConfig()

    def rank_pool(
        self,
        pool: List[ContextNode],
        query_embedding: np.ndarray,
        level: HierarchyLevel,
        similarity_threshold: float = None,
    ) -> List[RankedNode]:
        """
        Rank nodes within a single pool.

        Args:
            pool: List of context nodes in this pool
            query_embedding: Query embedding vector
            level: Hierarchy level of this pool
            similarity_threshold: Minimum similarity (for IMPLIED/BACKGROUND)

        Returns:
            List of RankedNode sorted by final score (descending)
        """
        if not pool:
            return []

        # Apply threshold for IMPLIED and BACKGROUND levels
        if similarity_threshold is None:
            if level == HierarchyLevel.IMPLIED:
                similarity_threshold = self.config.implied_threshold
            elif level == HierarchyLevel.BACKGROUND:
                similarity_threshold = self.config.background_threshold
            else:
                similarity_threshold = 0.0  # No threshold for STRUCTURAL/DIRECT

        ranked = []
        for node in pool:
            # Skip nodes without embeddings
            if node.embedding is None:
                continue

            # Calculate semantic similarity
            similarity = node.get_similarity(query_embedding)

            # Apply threshold
            if similarity < similarity_threshold:
                continue

            # Get within-level weight
            weight = node.within_level_weight

            # Combine scores
            # For STRUCTURAL/DIRECT: weight matters more (use 0.3 * sim + 0.7 * weight)
            # For IMPLIED/BACKGROUND: similarity matters more (use 0.7 * sim + 0.3 * weight)
            if level in [HierarchyLevel.STRUCTURAL, HierarchyLevel.DIRECT]:
                final_score = 0.3 * similarity + 0.7 * weight
            else:
                final_score = 0.7 * similarity + 0.3 * weight

            ranked.append(
                RankedNode(
                    node=node,
                    similarity_score=similarity,
                    weight_score=weight,
                    final_score=final_score,
                )
            )

        # Sort by final score (descending)
        ranked.sort(key=lambda x: x.final_score, reverse=True)

        return ranked

    def rank_all_pools(
        self,
        pools: Dict[HierarchyLevel, List[ContextNode]],
        query_embedding: np.ndarray,
    ) -> Dict[HierarchyLevel, List[RankedNode]]:
        """
        Rank nodes in all pools.

        Args:
            pools: Dictionary mapping HierarchyLevel to nodes
            query_embedding: Query embedding vector

        Returns:
            Dictionary mapping HierarchyLevel to ranked nodes
        """
        ranked_pools = {}

        for level, pool in pools.items():
            ranked_pools[level] = self.rank_pool(
                pool=pool,
                query_embedding=query_embedding,
                level=level,
            )

        return ranked_pools

    def select_within_budget(
        self,
        ranked: List[RankedNode],
        budget_tokens: int,
    ) -> List[ContextNode]:
        """
        Select top-ranked nodes that fit within budget.

        Args:
            ranked: Ranked nodes (sorted by score)
            budget_tokens: Token budget for this pool

        Returns:
            List of selected nodes (preserving rank order)
        """
        selected = []
        used_tokens = 0

        for ranked_node in ranked:
            node = ranked_node.node

            # Check if node fits in remaining budget
            if used_tokens + node.tokens <= budget_tokens:
                selected.append(node)
                used_tokens += node.tokens
            else:
                # Budget exceeded - stop selection
                break

        return selected

    def get_pool_summary(
        self,
        ranked_pool: List[RankedNode],
        budget_tokens: int,
    ) -> Dict[str, any]:
        """
        Get summary statistics for a ranked pool.

        Args:
            ranked_pool: Ranked nodes in pool
            budget_tokens: Token budget for pool

        Returns:
            Dictionary with summary stats
        """
        if not ranked_pool:
            return {
                "total_candidates": 0,
                "selected_count": 0,
                "budget_tokens": budget_tokens,
                "used_tokens": 0,
                "avg_similarity": 0.0,
                "avg_final_score": 0.0,
            }

        # Calculate what would be selected
        selected = self.select_within_budget(ranked_pool, budget_tokens)

        return {
            "total_candidates": len(ranked_pool),
            "selected_count": len(selected),
            "budget_tokens": budget_tokens,
            "used_tokens": sum(n.tokens for n in selected),
            "utilization": sum(n.tokens for n in selected) / budget_tokens if budget_tokens > 0 else 0,
            "avg_similarity": np.mean([r.similarity_score for r in ranked_pool]),
            "avg_final_score": np.mean([r.final_score for r in ranked_pool]),
            "top_similarity": ranked_pool[0].similarity_score if ranked_pool else 0.0,
            "bottom_similarity": ranked_pool[-1].similarity_score if ranked_pool else 0.0,
        }
