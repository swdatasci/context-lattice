"""
Feedback tracker: Learn from what context is actually used.

Detects which context nodes were referenced in responses and updates
usage counts, user boosts, and efficiency metrics.
"""

from typing import List, Optional, Set, Dict
import logging
import re
from datetime import datetime

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from ..core.node import ContextNode

logger = logging.getLogger(__name__)


class FeedbackTracker:
    """
    Track and learn from context usage.

    Features:
    - Detect which context was referenced in response
    - Increment usage counts
    - Apply user feedback boosts
    - Calculate efficiency metrics
    - Store in Redis for persistence
    """

    def __init__(
        self,
        redis_url: str = "redis://10.32.3.27:6379",
        ttl: int = 2592000,  # 30 days
        key_prefix: str = "context-lattice:feedback:",
        enabled: bool = True,
    ):
        """
        Initialize feedback tracker.

        Args:
            redis_url: Redis connection URL
            ttl: Time-to-live for feedback data (seconds)
            key_prefix: Redis key prefix
            enabled: Whether tracking is enabled
        """
        self.enabled = enabled and REDIS_AVAILABLE
        self.ttl = ttl
        self.key_prefix = key_prefix

        if self.enabled:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info(f"Feedback tracker enabled: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, tracking disabled")
                self.enabled = False

    def track_usage(
        self,
        query: str,
        response: str,
        context_provided: List[ContextNode],
    ) -> Dict[str, any]:
        """
        Track which context was used in response.

        Args:
            query: Original query
            response: Generated response
            context_provided: Context nodes provided to LLM

        Returns:
            Dictionary with usage statistics
        """
        if not context_provided:
            return {
                "referenced_count": 0,
                "provided_count": 0,
                "efficiency": 0.0,
                "referenced_nodes": [],
            }

        # Detect which nodes were referenced
        referenced_ids = []
        for node in context_provided:
            if self._was_referenced(node, response):
                referenced_ids.append(node.id)
                # Increment usage count
                self._increment_usage(node.id)

        # Calculate efficiency
        efficiency = len(referenced_ids) / len(context_provided) if context_provided else 0.0

        # Log usage record
        self._log_usage_record(query, referenced_ids, len(context_provided))

        stats = {
            "referenced_count": len(referenced_ids),
            "provided_count": len(context_provided),
            "efficiency": efficiency,
            "referenced_nodes": referenced_ids,
        }

        logger.info(
            f"Context efficiency: {efficiency:.1%} "
            f"({len(referenced_ids)}/{len(context_provided)} used)"
        )

        return stats

    def _was_referenced(self, node: ContextNode, response: str) -> bool:
        """
        Detect if node was referenced in response.

        Args:
            node: Context node
            response: Generated response

        Returns:
            True if node was referenced
        """
        # Check for file path mentions
        if file_path := node.metadata.get('file_path'):
            if file_path in response:
                return True
            # Also check just the filename
            from pathlib import Path
            if Path(file_path).name in response:
                return True

        # Check for entity name mentions
        if entity_name := node.metadata.get('entity_name'):
            # Word boundary matching to avoid partial matches
            pattern = r'\b' + re.escape(entity_name) + r'\b'
            if re.search(pattern, response):
                return True

        # Check for code snippet overlap (simplified)
        if node.level.value <= 1:  # STRUCTURAL or DIRECT
            # For code nodes, check for significant overlap
            node_words = set(node.content.lower().split())
            response_words = set(response.lower().split())
            overlap = node_words & response_words

            # If >20% of node words appear in response, consider it referenced
            if len(overlap) / len(node_words) > 0.2 if node_words else False:
                return True

        return False

    def _increment_usage(self, node_id: str):
        """Increment usage count for node."""
        if not self.enabled:
            return

        try:
            key = f"{self.key_prefix}usage:{node_id}"
            self.redis_client.incr(key)
            self.redis_client.expire(key, self.ttl)
        except Exception as e:
            logger.warning(f"Failed to increment usage for {node_id}: {e}")

    def get_usage_count(self, node_id: str) -> int:
        """Get usage count for node."""
        if not self.enabled:
            return 0

        try:
            key = f"{self.key_prefix}usage:{node_id}"
            count = self.redis_client.get(key)
            return int(count) if count else 0
        except Exception as e:
            logger.warning(f"Failed to get usage for {node_id}: {e}")
            return 0

    def apply_user_feedback(
        self,
        node_id: str,
        feedback_type: str,
        feedback_value: Optional[str] = None,
    ):
        """
        Apply user feedback to node.

        Args:
            node_id: Node ID
            feedback_type: Type of feedback ('helpful', 'not_helpful', 'correction')
            feedback_value: Optional correction text
        """
        if not self.enabled:
            return

        try:
            if feedback_type == "helpful":
                # Boost this node
                key = f"{self.key_prefix}boost:{node_id}"
                self.redis_client.set(key, "1.5", ex=self.ttl)
                logger.info(f"Applied helpful boost to {node_id}")

            elif feedback_type == "not_helpful":
                # Penalize this node
                key = f"{self.key_prefix}boost:{node_id}"
                self.redis_client.set(key, "0.7", ex=self.ttl)
                logger.info(f"Applied not-helpful penalty to {node_id}")

            elif feedback_type == "correction" and feedback_value:
                # Store correction
                key = f"{self.key_prefix}correction:{node_id}"
                correction_data = {
                    "original_id": node_id,
                    "correction": feedback_value,
                    "timestamp": datetime.now().isoformat(),
                }
                self.redis_client.set(key, str(correction_data), ex=self.ttl)
                # Also boost the node
                boost_key = f"{self.key_prefix}boost:{node_id}"
                self.redis_client.set(boost_key, "1.5", ex=self.ttl)
                logger.info(f"Stored correction for {node_id}")

        except Exception as e:
            logger.warning(f"Failed to apply feedback for {node_id}: {e}")

    def get_user_boost(self, node_id: str) -> float:
        """Get user feedback boost for node."""
        if not self.enabled:
            return 1.0

        try:
            key = f"{self.key_prefix}boost:{node_id}"
            boost = self.redis_client.get(key)
            return float(boost) if boost else 1.0
        except Exception as e:
            logger.warning(f"Failed to get boost for {node_id}: {e}")
            return 1.0

    def _log_usage_record(
        self,
        query: str,
        referenced_ids: List[str],
        total_provided: int,
    ):
        """Log usage record for analytics."""
        if not self.enabled:
            return

        try:
            key = f"{self.key_prefix}records:{datetime.now().strftime('%Y%m%d')}"
            record = {
                "timestamp": datetime.now().isoformat(),
                "query": query[:100],  # Truncate long queries
                "referenced_count": len(referenced_ids),
                "total_count": total_provided,
                "efficiency": len(referenced_ids) / total_provided if total_provided else 0,
            }
            # Store as list (LPUSH)
            self.redis_client.lpush(key, str(record))
            self.redis_client.expire(key, self.ttl)
        except Exception as e:
            logger.warning(f"Failed to log usage record: {e}")

    def get_efficiency_stats(self, days: int = 7) -> Dict[str, float]:
        """
        Get efficiency statistics over last N days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with efficiency stats
        """
        if not self.enabled:
            return {
                "avg_efficiency": 0.0,
                "total_queries": 0,
                "avg_referenced": 0.0,
                "avg_provided": 0.0,
            }

        try:
            total_efficiency = 0.0
            total_queries = 0
            total_referenced = 0
            total_provided = 0

            # Scan records from last N days
            from datetime import timedelta
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
                key = f"{self.key_prefix}records:{date}"

                records = self.redis_client.lrange(key, 0, -1)
                for record_str in records:
                    # Parse record (simplified - in production use JSON)
                    if "efficiency" in record_str:
                        total_queries += 1
                        # Extract efficiency value (simplified parsing)
                        # In production, use proper JSON serialization

            # Calculate averages
            avg_efficiency = total_efficiency / total_queries if total_queries else 0.0
            avg_referenced = total_referenced / total_queries if total_queries else 0.0
            avg_provided = total_provided / total_queries if total_queries else 0.0

            return {
                "avg_efficiency": avg_efficiency,
                "total_queries": total_queries,
                "avg_referenced": avg_referenced,
                "avg_provided": avg_provided,
            }

        except Exception as e:
            logger.warning(f"Failed to get efficiency stats: {e}")
            return {
                "avg_efficiency": 0.0,
                "total_queries": 0,
                "avg_referenced": 0.0,
                "avg_provided": 0.0,
            }

    def enrich_nodes(self, nodes: List[ContextNode]) -> List[ContextNode]:
        """
        Enrich nodes with feedback data (usage counts, boosts).

        Args:
            nodes: List of context nodes

        Returns:
            Enriched nodes with updated usage_count and user_boost
        """
        if not self.enabled:
            return nodes

        for node in nodes:
            # Get usage count from Redis
            usage_count = self.get_usage_count(node.id)
            if usage_count > 0:
                node.usage_count = usage_count

            # Get user boost from Redis
            user_boost = self.get_user_boost(node.id)
            if user_boost != 1.0:
                node.user_boost = user_boost

        return nodes
