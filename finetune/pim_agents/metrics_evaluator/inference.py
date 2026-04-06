"""
Fast inference for Metrics Evaluator classification.

Classifies committee signals (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL).

Usage:
    from finetune.pim_agents.metrics_evaluator.inference import SignalClassifier

    classifier = SignalClassifier("models/metrics_evaluator/final")
    result = classifier.classify("Committee vote: 6/7 agents recommend LONG")
    print(result.signal, result.confidence)
"""

import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Signal(Enum):
    """Trading signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class SignalClassification:
    """Result of signal classification."""
    signal: Signal
    confidence: float
    all_scores: dict  # Scores for all signal types


class SignalClassifier:
    """
    Fine-tuned signal classifier for PIM.

    Supports PyTorch and ONNX backends.
    """

    # Default to Hugging Face Hub model
    DEFAULT_MODEL = "zkarbie/pim-metrics-evaluator"

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

    def classify(self, description: str) -> SignalClassification:
        """Classify signal from committee description."""
        if self.onnx_session is not None:
            return self._classify_onnx(description)
        else:
            return self._classify_pytorch(description)

    def _classify_onnx(self, description: str) -> SignalClassification:
        """Classify using ONNX runtime."""
        inputs = self.tokenizer(
            description,
            return_tensors="np",
            padding="max_length",
            max_length=128,
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

        label = self.id2label.get(pred_id, "neutral")
        signal = Signal(label) if label in [s.value for s in Signal] else Signal.NEUTRAL

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return SignalClassification(
            signal=signal,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _classify_pytorch(self, description: str) -> SignalClassification:
        """Classify using PyTorch."""
        import torch

        inputs = self.tokenizer(
            description,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        confidence = float(probs[pred_id])

        label = self.id2label.get(pred_id, "neutral")
        signal = Signal(label) if label in [s.value for s in Signal] else Signal.NEUTRAL

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return SignalClassification(
            signal=signal,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def classify_from_votes(
        self,
        votes_for: int,
        votes_against: int,
        total_agents: int,
        confidence: float,
    ) -> SignalClassification:
        """
        Classify signal from vote counts.

        Args:
            votes_for: Number of agents recommending LONG
            votes_against: Number of agents recommending SHORT
            total_agents: Total number of agents
            confidence: Average confidence

        Returns:
            SignalClassification
        """
        neutral = total_agents - votes_for - votes_against
        description = (
            f"Committee vote: {votes_for}/{total_agents} agents recommend LONG, "
            f"{votes_against} recommend SHORT, {neutral} HOLD, "
            f"average confidence {confidence:.0%}"
        )
        return self.classify(description)


def test_classifier(model_path: str):
    """Test the classifier with sample descriptions."""
    classifier = SignalClassifier(model_path)

    test_cases = [
        "Committee vote: 6/7 agents recommend LONG, average confidence 85%",
        "Moderate bullish: 4/7 agents recommend LONG, confidence 60%",
        "Mixed signals: 3 LONG, 3 SHORT, 1 HOLD at 45%",
        "Committee leans SHORT: 4 agents agree at 62% confidence",
        "Strong consensus: RiskAgent, TrendAgent, MacroAgent all bearish with high confidence 88%",
        "No consensus: agents split evenly, low confidence 40%",
    ]

    print("\nTesting signal classifier:")
    print("-" * 70)

    for description in test_cases:
        result = classifier.classify(description)
        print(f"Input: {description[:55]}...")
        print(f"  Signal: {result.signal.value}, Confidence: {result.confidence:.3f}")
        print()


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else SignalClassifier.DEFAULT_MODEL
    test_classifier(model_path)
