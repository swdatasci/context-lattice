"""
Semantic search source: Integrates with Caelum's vector database.

Fetches context from Qdrant using existing triple-model ensemble embeddings.
"""

from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import numpy as np

from ..core.node import ContextNode
from ..core.hierarchy import HierarchyLevel

logger = logging.getLogger(__name__)


class SemanticSource:
    """
    Fetch context from Qdrant vector database.

    Uses Caelum's existing embeddings (12,481+ vectors across 1,039+ files).
    """

    def __init__(
        self,
        qdrant_url: str = "http://10.32.3.27:6333",
        collection: str = "caelum_knowledge",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize semantic source.

        Args:
            qdrant_url: Qdrant server URL
            collection: Collection name
            model_name: Sentence transformer model for query embedding
        """
        self.qdrant_url = qdrant_url
        self.collection = collection

        try:
            self.client = QdrantClient(url=qdrant_url)
            logger.info(f"Connected to Qdrant at {qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None

        # Model for query embedding
        self.model = SentenceTransformer(model_name)

    def fetch(
        self,
        query: str,
        limit: int = 10,
        project: Optional[str] = None,
        since: Optional[datetime] = None,
        type_filter: Optional[str] = None,
    ) -> List[ContextNode]:
        """
        Fetch context nodes from semantic search.

        Args:
            query: Search query
            limit: Maximum results
            project: Filter by project name (e.g., "PassiveIncomeMaximizer")
            since: Only results since date
            type_filter: Filter by document type (e.g., "architecture", "session")

        Returns:
            List of ContextNode with embeddings from Qdrant
        """
        if not self.client:
            logger.warning("Qdrant client not available, returning empty")
            return []

        try:
            # Embed query
            query_vector = self.model.encode(query).tolist()

            # Build filter
            filter_conditions = []
            if project:
                filter_conditions.append(
                    FieldCondition(key="project", match=MatchValue(value=project))
                )
            if type_filter:
                filter_conditions.append(
                    FieldCondition(key="type", match=MatchValue(value=type_filter))
                )
            # Note: Date filtering would require indexed_at field in Qdrant

            search_filter = Filter(must=filter_conditions) if filter_conditions else None

            # Search Qdrant
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
            )

            # Convert to ContextNodes
            nodes = []
            for i, hit in enumerate(results):
                node = self._result_to_node(hit, rank=i)
                if node:
                    nodes.append(node)

            logger.info(f"Semantic search returned {len(nodes)} nodes for query: {query[:50]}...")
            return nodes

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _result_to_node(self, hit: Any, rank: int) -> Optional[ContextNode]:
        """
        Convert Qdrant search result to ContextNode.

        Args:
            hit: Qdrant search result
            rank: Result rank (0-based)

        Returns:
            ContextNode or None if conversion fails
        """
        try:
            payload = hit.payload
            score = hit.score

            # Extract content
            content = payload.get("content", payload.get("text", ""))
            if not content:
                return None

            # Extract metadata
            file_path = payload.get("file_path")
            chunk_index = payload.get("chunk_index", 0)
            header = payload.get("header", "")
            doc_type = payload.get("type", "unknown")

            # Generate node ID
            node_id = f"semantic_{file_path}_{chunk_index}" if file_path else f"semantic_{rank}"

            # Estimate tokens (4 chars per token)
            tokens = len(content) // 4 + 1

            # Determine hierarchy level based on doc type
            level = self._determine_level(doc_type, file_path)

            # Get timestamp (if available)
            timestamp = payload.get("indexed_at")
            if timestamp and isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    timestamp = None

            # Get embedding vector (if available in payload, otherwise use search vector)
            embedding = np.array(hit.vector) if hasattr(hit, 'vector') and hit.vector else None

            # Create metadata
            metadata = {
                "file_path": file_path,
                "chunk_index": chunk_index,
                "header": header,
                "type": doc_type,
                "similarity_score": score,
                "rank": rank,
                "source": "semantic_search",
            }

            return ContextNode(
                id=node_id,
                content=content,
                tokens=tokens,
                level=level,
                embedding=embedding,
                metadata=metadata,
                timestamp=timestamp,
            )

        except Exception as e:
            logger.error(f"Failed to convert search result to node: {e}")
            return None

    def _determine_level(self, doc_type: str, file_path: Optional[str]) -> HierarchyLevel:
        """
        Determine hierarchy level based on document type and file path.

        Args:
            doc_type: Document type (architecture, session, guide, etc.)
            file_path: File path (if available)

        Returns:
            HierarchyLevel assignment
        """
        # STRUCTURAL: Project context, user preferences
        if file_path and any(name in file_path.lower() for name in ["claude.md", "user_pref"]):
            return HierarchyLevel.STRUCTURAL

        # DIRECT: Recent session notes, guides
        if doc_type in ["session", "guide"]:
            return HierarchyLevel.DIRECT

        # BACKGROUND: Architecture, research, general docs
        if doc_type in ["architecture", "research", "documentation"]:
            return HierarchyLevel.BACKGROUND

        # IMPLIED: Everything else (to be refined by pool selector)
        return HierarchyLevel.IMPLIED

    def test_connection(self) -> bool:
        """
        Test Qdrant connection.

        Returns:
            True if connected, False otherwise
        """
        if not self.client:
            return False

        try:
            collections = self.client.get_collections()
            logger.info(f"Qdrant connection OK: {len(collections.collections)} collections")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection test failed: {e}")
            return False
