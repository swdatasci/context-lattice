"""
Multi-source collector: Orchestrates context fetching from multiple sources.

Handles parallel fetching, caching, deduplication, and graceful degradation.
"""

from typing import List, Optional, Dict, Set
import logging
import hashlib
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from ..core.node import ContextNode
from .semantic_source import SemanticSource
from .file_source import FileSource

logger = logging.getLogger(__name__)


class MultiSourceCollector:
    """
    Collect context from multiple sources in parallel.

    Features:
    - Parallel fetching
    - Redis caching
    - Graceful degradation (continue if source fails)
    - Deduplication
    """

    def __init__(
        self,
        semantic_config: Optional[Dict] = None,
        file_config: Optional[Dict] = None,
        cache_config: Optional[Dict] = None,
    ):
        """
        Initialize multi-source collector.

        Args:
            semantic_config: Configuration for semantic source
            file_config: Configuration for file source
            cache_config: Configuration for caching (Redis)
        """
        # Initialize sources
        semantic_config = semantic_config or {}
        file_config = file_config or {}

        self.semantic_enabled = semantic_config.get('enabled', False)  # Default False to avoid Qdrant hangs
        self.file_enabled = file_config.get('enabled', True)

        if self.semantic_enabled:
            try:
                self.semantic_source = SemanticSource(
                    qdrant_url=semantic_config.get('qdrant_url', 'http://10.32.3.27:6333'),
                    collection=semantic_config.get('collection', 'caelum_knowledge'),
                )
            except Exception as e:
                logger.warning(f"Failed to initialize semantic source: {e}")
                self.semantic_enabled = False

        if self.file_enabled:
            try:
                self.file_source = FileSource(
                    file_types=file_config.get('file_types'),
                    exclude_dirs=file_config.get('exclude_dirs'),
                )
            except Exception as e:
                logger.warning(f"Failed to initialize file source: {e}")
                self.file_enabled = False

        # Initialize cache
        cache_config = cache_config or {}
        self.cache_enabled = cache_config.get('enabled', True) and REDIS_AVAILABLE

        if self.cache_enabled:
            try:
                redis_url = cache_config.get('redis_url', 'redis://10.32.3.27:6379')
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                self.cache_ttl = cache_config.get('ttl', 3600)
                self.cache_prefix = cache_config.get('key_prefix', 'context-lattice:')
                logger.info(f"Redis cache enabled: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}, caching disabled")
                self.cache_enabled = False

    def collect(
        self,
        query: str,
        project_root: Optional[Path] = None,
        current_file: Optional[str] = None,
        sources: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> List[ContextNode]:
        """
        Collect context from all enabled sources.

        Args:
            query: Search query
            project_root: Project root directory (for file source)
            current_file: Currently open file
            sources: Specific sources to use (default: all enabled)
            use_cache: Whether to use cache

        Returns:
            List of ContextNode from all sources
        """
        # Check cache first
        if use_cache and self.cache_enabled:
            cached = self._get_cached(query, project_root, sources)
            if cached:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached

        # Determine which sources to use
        active_sources = sources or []
        if not active_sources:
            if self.semantic_enabled:
                active_sources.append('semantic')
            if self.file_enabled:
                active_sources.append('file')

        # Collect from sources in parallel
        nodes = self._collect_parallel(
            query=query,
            project_root=project_root,
            current_file=current_file,
            sources=active_sources,
        )

        # Deduplicate
        nodes = self._deduplicate(nodes)

        # Cache results
        if use_cache and self.cache_enabled:
            self._set_cached(query, project_root, sources, nodes)

        logger.info(f"Collected {len(nodes)} nodes from {len(active_sources)} sources")
        return nodes

    def _collect_parallel(
        self,
        query: str,
        project_root: Optional[Path],
        current_file: Optional[str],
        sources: List[str],
    ) -> List[ContextNode]:
        """
        Collect from sources in parallel using thread pool.

        Args:
            query: Search query
            project_root: Project root
            current_file: Current file
            sources: Sources to collect from

        Returns:
            Combined list of nodes
        """
        all_nodes = []

        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            futures = []

            # Submit tasks
            if 'semantic' in sources and self.semantic_enabled:
                future = executor.submit(self._fetch_semantic, query)
                futures.append(('semantic', future))

            if 'file' in sources and self.file_enabled:
                if project_root:
                    future = executor.submit(
                        self._fetch_files,
                        query,
                        project_root,
                        current_file,
                    )
                    futures.append(('file', future))

            # Collect results
            for source_name, future in futures:
                try:
                    nodes = future.result(timeout=10)  # 10s timeout per source
                    all_nodes.extend(nodes)
                    logger.info(f"Source '{source_name}' returned {len(nodes)} nodes")
                except Exception as e:
                    logger.error(f"Source '{source_name}' failed: {e}")
                    # Continue with other sources (graceful degradation)

        return all_nodes

    def _fetch_semantic(self, query: str) -> List[ContextNode]:
        """Fetch from semantic source."""
        try:
            return self.semantic_source.fetch(query, limit=10)
        except Exception as e:
            logger.error(f"Semantic fetch failed: {e}")
            return []

    def _fetch_files(
        self,
        query: str,
        project_root: Path,
        current_file: Optional[str],
    ) -> List[ContextNode]:
        """Fetch from file source."""
        try:
            return self.file_source.fetch(
                query=query,
                project_root=project_root,
                current_file=current_file,
                max_files=20,
            )
        except Exception as e:
            logger.error(f"File fetch failed: {e}")
            return []

    def _deduplicate(self, nodes: List[ContextNode]) -> List[ContextNode]:
        """
        Deduplicate nodes by content.

        Args:
            nodes: List of nodes (may have duplicates)

        Returns:
            Deduplicated list
        """
        seen_content = set()
        unique_nodes = []

        for node in nodes:
            # Create hash of content
            content_hash = hashlib.md5(node.content.encode()).hexdigest()

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_nodes.append(node)

        if len(nodes) != len(unique_nodes):
            logger.info(f"Deduplicated: {len(nodes)} -> {len(unique_nodes)} nodes")

        return unique_nodes

    def _get_cached(
        self,
        query: str,
        project_root: Optional[Path],
        sources: Optional[List[str]],
    ) -> Optional[List[ContextNode]]:
        """Get cached results."""
        if not self.cache_enabled:
            return None

        try:
            cache_key = self._make_cache_key(query, project_root, sources)
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                # Deserialize nodes
                nodes_data = json.loads(cached_data)
                nodes = [self._deserialize_node(n) for n in nodes_data]
                return nodes

        except Exception as e:
            logger.warning(f"Cache get failed: {e}")

        return None

    def _set_cached(
        self,
        query: str,
        project_root: Optional[Path],
        sources: Optional[List[str]],
        nodes: List[ContextNode],
    ):
        """Set cached results."""
        if not self.cache_enabled:
            return

        try:
            cache_key = self._make_cache_key(query, project_root, sources)

            # Serialize nodes (without embeddings to save space)
            nodes_data = [self._serialize_node(n) for n in nodes]
            cached_data = json.dumps(nodes_data)

            self.redis_client.setex(cache_key, self.cache_ttl, cached_data)

        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def _make_cache_key(
        self,
        query: str,
        project_root: Optional[Path],
        sources: Optional[List[str]],
    ) -> str:
        """Generate cache key."""
        key_parts = [
            query,
            str(project_root) if project_root else "",
            ",".join(sorted(sources)) if sources else "",
        ]
        key_str = "|".join(key_parts)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{self.cache_prefix}query:{key_hash}"

    def _serialize_node(self, node: ContextNode) -> Dict:
        """Serialize node for caching (without embedding)."""
        return {
            "id": node.id,
            "content": node.content,
            "tokens": node.tokens,
            "level": node.level.value,
            "recency_score": node.recency_score,
            "usage_count": node.usage_count,
            "user_boost": node.user_boost,
            "metadata": node.metadata,
            "timestamp": node.timestamp.isoformat() if node.timestamp else None,
        }

    def _deserialize_node(self, data: Dict) -> ContextNode:
        """Deserialize node from cache."""
        from ..core.hierarchy import HierarchyLevel
        from datetime import datetime

        return ContextNode(
            id=data["id"],
            content=data["content"],
            tokens=data["tokens"],
            level=HierarchyLevel(data["level"]),
            embedding=None,  # Embeddings not cached
            recency_score=data["recency_score"],
            usage_count=data["usage_count"],
            user_boost=data["user_boost"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data["timestamp"] else None,
        )
