"""
Fast inference for Semantic Reranker.

Reranks search results based on query-document relevance.

Usage:
    from finetune.semantic_reranker.inference import SemanticReranker

    reranker = SemanticReranker("models/semantic_reranker/final")
    results = reranker.rerank("How to implement caching", documents)
    print(results)  # Sorted by relevance
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

from sentence_transformers import CrossEncoder


@dataclass
class RerankedResult:
    """A reranked search result."""
    document: str
    score: float
    original_index: int
    metadata: Optional[dict] = None


class SemanticReranker:
    """
    Fine-tuned semantic reranker using cross-encoder.

    Reranks search results based on query-document relevance.
    """

    # Default to Hugging Face Hub model
    DEFAULT_MODEL = "zkarbie/context-lattice-semantic-reranker"

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
    ):
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL

        # Load model (supports HF Hub ID or local path)
        model_id = str(self.model_path) if isinstance(self.model_path, Path) else self.model_path
        self.model = CrossEncoder(model_id, max_length=256)
        if device:
            self.model.model.to(device)

        print(f"Loaded reranker from {model_id}")

        # Load label info if available (local path only)
        self.label_info = {}
        if isinstance(self.model_path, Path):
            label_info_path = self.model_path / "label_info.json"
            if label_info_path.exists():
                with open(label_info_path) as f:
                    self.label_info = json.load(f)

    def score(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.

        Returns:
            Relevance score (0-1, higher is more relevant)
        """
        return float(self.model.predict([[query, document]])[0])

    def score_batch(self, query: str, documents: List[str]) -> List[float]:
        """
        Score multiple documents for a single query.

        Returns:
            List of relevance scores
        """
        pairs = [[query, doc] for doc in documents]
        return [float(s) for s in self.model.predict(pairs)]

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
        threshold: float = None,
        metadata: List[dict] = None,
    ) -> List[RerankedResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Return only top K results (default: all)
            threshold: Return only results above this score (default: none)
            metadata: Optional metadata for each document

        Returns:
            List of RerankedResult sorted by relevance (highest first)
        """
        if not documents:
            return []

        # Score all documents
        scores = self.score_batch(query, documents)

        # Create results with original indices
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            meta = metadata[i] if metadata and i < len(metadata) else None
            results.append(RerankedResult(
                document=doc,
                score=score,
                original_index=i,
                metadata=meta
            ))

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply threshold filter
        if threshold is not None:
            results = [r for r in results if r.score >= threshold]

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_with_context(
        self,
        query: str,
        results: List[dict],
        content_key: str = "content",
        top_k: int = None,
    ) -> List[dict]:
        """
        Rerank search results that include metadata.

        Args:
            query: The search query
            results: List of dicts with content and metadata
            content_key: Key for the document content in each result
            top_k: Return only top K results

        Returns:
            Reranked results with added 'rerank_score' field
        """
        if not results:
            return []

        documents = [r[content_key] for r in results]
        scores = self.score_batch(query, documents)

        # Add scores and sort
        for result, score in zip(results, scores):
            result['rerank_score'] = score

        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results


def test_reranker(model_path: str):
    """Test the reranker with sample data."""
    reranker = SemanticReranker(model_path)

    query = "How to implement caching in Python"

    documents = [
        "Caching is a technique to store frequently accessed data in memory. "
        "In Python, you can use functools.lru_cache for memoization or Redis for distributed caching.",

        "Company holiday schedule: Office closed Dec 25-Jan 1. "
        "Please submit PTO requests at least 2 weeks in advance.",

        "Python is a programming language. It was created by Guido van Rossum.",

        "Best practices for caching: Set appropriate TTLs, handle cache invalidation, "
        "use cache-aside pattern. Example implementation with Redis and Python included.",
    ]

    print(f"\nQuery: {query}")
    print("-" * 70)

    results = reranker.rerank(query, documents)

    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result.score:.3f} (original index: {result.original_index})")
        print(f"   {result.document[:80]}...")


if __name__ == "__main__":
    import sys

    # Default to HF Hub model, or use local path if provided
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_reranker(model_path or SemanticReranker.DEFAULT_MODEL)
