"""
Fast inference for fine-tuned Intent Classifier.

Supports both PyTorch and ONNX backends.
Drop-in replacement for rule-based IntentClassifier.

Usage:
    from finetune.intent_classifier.inference import FineTunedIntentClassifier

    classifier = FineTunedIntentClassifier("models/intent_classifier/final")
    result = classifier.classify("Fix the bug in login.py")
    print(result.intent, result.confidence)
"""

import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

import numpy as np


class QueryIntent(Enum):
    """Query intent categories (matches rule-based classifier)."""
    CODING = "CODING"
    DEBUGGING = "DEBUGGING"
    REFACTORING = "REFACTORING"
    RESEARCH = "RESEARCH"
    PLANNING = "PLANNING"
    DOCUMENTATION = "DOCUMENTATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class IntentMatch:
    """Result of intent classification (matches rule-based classifier)."""
    intent: QueryIntent
    confidence: float
    matched_patterns: List[str]  # Empty for fine-tuned model


class FineTunedIntentClassifier:
    """
    Fine-tuned DistilBERT intent classifier.

    Drop-in replacement for rule-based IntentClassifier.
    Supports PyTorch and ONNX backends for inference.
    """

    # Default to Hugging Face Hub model
    DEFAULT_MODEL = "zkarbie/context-lattice-intent-classifier"

    def __init__(
        self,
        model_path: str = None,
        use_onnx: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize classifier.

        Args:
            model_path: Path to saved model directory or HF Hub ID (default: HF Hub)
            use_onnx: Use ONNX runtime (faster) if available
            device: PyTorch device ("cpu" or "cuda")
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL
        self.device = device
        self.use_onnx = use_onnx

        # Load label mapping
        with open(self.model_path / "label_mapping.json") as f:
            mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
            self.label2id = mapping["label2id"]

        # Try ONNX first, fallback to PyTorch
        self.onnx_session = None
        self.model = None
        self.tokenizer = None

        if use_onnx and (self.model_path / "model.onnx").exists():
            self._load_onnx()
        else:
            self._load_pytorch()

    def _load_onnx(self):
        """Load ONNX model for fast inference."""
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

    def classify(self, query: str) -> IntentMatch:
        """
        Classify query intent.

        Args:
            query: User query string

        Returns:
            IntentMatch with intent and confidence
        """
        if self.onnx_session is not None:
            return self._classify_onnx(query)
        else:
            return self._classify_pytorch(query)

    def _classify_onnx(self, query: str) -> IntentMatch:
        """Classify using ONNX runtime."""
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="np",
            padding="max_length",
            max_length=128,
            truncation=True,
        )

        # Run inference
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

        label = self.id2label.get(pred_id, "UNKNOWN")
        intent = QueryIntent[label] if label in QueryIntent.__members__ else QueryIntent.UNKNOWN

        return IntentMatch(
            intent=intent,
            confidence=confidence,
            matched_patterns=[],
        )

    def _classify_pytorch(self, query: str) -> IntentMatch:
        """Classify using PyTorch."""
        import torch

        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        confidence = float(probs[pred_id])

        label = self.id2label.get(pred_id, "UNKNOWN")
        intent = QueryIntent[label] if label in QueryIntent.__members__ else QueryIntent.UNKNOWN

        return IntentMatch(
            intent=intent,
            confidence=confidence,
            matched_patterns=[],
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def classify_simple(self, query: str) -> QueryIntent:
        """Simplified classification returning only intent."""
        return self.classify(query).intent

    def get_intent_name(self, query: str) -> str:
        """Get intent as string (for HierarchyConfig compatibility)."""
        return self.classify_simple(query).value


# Convenience function for quick testing
def test_classifier(model_path: str):
    """Test the classifier with sample queries."""
    classifier = FineTunedIntentClassifier(model_path)

    test_queries = [
        "Fix the authentication bug in login.py",
        "How does the context-lattice hierarchy work?",
        "Add a new endpoint for user registration",
        "Refactor the database connection code",
        "Plan the Phase 3 implementation",
        "Document the API endpoints",
    ]

    print("\nTesting classifier:")
    print("-" * 60)

    for query in test_queries:
        result = classifier.classify(query)
        print(f"Query: {query}")
        print(f"  Intent: {result.intent.value}, Confidence: {result.confidence:.3f}")
        print()


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else FineTunedIntentClassifier.DEFAULT_MODEL
    test_classifier(model_path)
