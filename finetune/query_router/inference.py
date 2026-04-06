"""
Fast inference for Query Router.

Routes queries to appropriate Caelum services.

Usage:
    from finetune.query_router.inference import QueryRouter

    router = QueryRouter("models/query_router/final")
    result = router.route("Search for authentication docs")
    print(result.route, result.confidence)
"""

import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Route(Enum):
    """Query routing categories."""
    SEMANTIC_SEARCH = "semantic_search"
    FILE_OPS = "file_ops"
    TRADING = "trading"
    CODE_ANALYSIS = "code_analysis"
    DATABASE = "database"
    GIT_OPS = "git_ops"
    GENERAL = "general"


@dataclass
class RouteMatch:
    """Result of query routing."""
    route: Route
    confidence: float
    all_scores: dict  # Scores for all routes


class QueryRouter:
    """
    Fine-tuned query router for Caelum services.

    Supports PyTorch and ONNX backends.
    """

    # Default to Hugging Face Hub model
    DEFAULT_MODEL = "zkarbie/context-lattice-query-router"

    def __init__(
        self,
        model_path: str = None,
        use_onnx: bool = True,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL
        self.device = device
        self.use_onnx = use_onnx

        with open(self.model_path / "label_mapping.json") as f:
            mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
            self.label2id = mapping["label2id"]

        self.onnx_session = None
        self.model = None
        self.tokenizer = None

        if use_onnx and (self.model_path / "model.onnx").exists():
            self._load_onnx()
        else:
            self._load_pytorch()

    def _load_onnx(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            from transformers import DistilBertTokenizer

            onnx_path = self.model_path / "model.onnx"
            self.onnx_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.model_path))
            print(f"Loaded ONNX model from {onnx_path}")

        except ImportError:
            print("ONNX runtime not available, falling back to PyTorch")
            self._load_pytorch()

    def _load_pytorch(self):
        """Load PyTorch model."""
        from transformers import (
            DistilBertTokenizer,
            DistilBertForSequenceClassification,
        )
        import torch

        self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.model_path))
        self.model = DistilBertForSequenceClassification.from_pretrained(
            str(self.model_path)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded PyTorch model from {self.model_path}")

    def route(self, query: str) -> RouteMatch:
        """Route a query to appropriate service."""
        if self.onnx_session is not None:
            return self._route_onnx(query)
        else:
            return self._route_pytorch(query)

    def _route_onnx(self, query: str) -> RouteMatch:
        """Route using ONNX runtime."""
        inputs = self.tokenizer(
            query,
            return_tensors="np",
            padding="max_length",
            max_length=64,
            truncation=True,
        )

        outputs = self.onnx_session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )

        logits = outputs[0]
        probs = self._softmax(logits[0])
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])

        label = self.id2label.get(pred_id, "general")
        route = Route(label) if label in [r.value for r in Route] else Route.GENERAL

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return RouteMatch(
            route=route,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _route_pytorch(self, query: str) -> RouteMatch:
        """Route using PyTorch."""
        import torch

        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        confidence = float(probs[pred_id])

        label = self.id2label.get(pred_id, "general")
        route = Route(label) if label in [r.value for r in Route] else Route.GENERAL

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return RouteMatch(
            route=route,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def route_simple(self, query: str) -> str:
        """Return route name string."""
        return self.route(query).route.value


def test_router(model_path: str):
    """Test the router with sample queries."""
    router = QueryRouter(model_path)

    test_queries = [
        "Search for authentication documentation",
        "Read the CLAUDE.md file",
        "What's the current portfolio status?",
        "Debug the connection error",
        "What tables store trading data?",
        "Commit these changes",
        "What's next on our todo list?",
    ]

    print("\nTesting router:")
    print("-" * 70)

    for query in test_queries:
        result = router.route(query)
        print(f"Query: {query}")
        print(f"  Route: {result.route.value}, Confidence: {result.confidence:.3f}")
        print()


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else QueryRouter.DEFAULT_MODEL
    test_router(model_path)
