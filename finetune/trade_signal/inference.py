"""
Fast inference for Trade Signal classification.

Classifies FinColl predictions into trading actions.

Usage:
    from finetune.trade_signal.inference import TradeSignalClassifier

    classifier = TradeSignalClassifier("models/trade_signal/final")
    result = classifier.classify("FinColl predicts bullish with 80% confidence")
    print(result.action, result.confidence)
"""

import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np


class TradeAction(Enum):
    """Trading action types."""
    STRONG_LONG = "strong_long"
    LONG = "long"
    HOLD = "hold"
    SHORT = "short"
    STRONG_SHORT = "strong_short"
    AVOID = "avoid"


@dataclass
class TradeSignalResult:
    """Result of trade signal classification."""
    action: TradeAction
    confidence: float
    all_scores: dict


class TradeSignalClassifier:
    """
    Fine-tuned trade signal classifier.

    Supports PyTorch and ONNX backends.
    """

    # Default to Hugging Face Hub model
    DEFAULT_MODEL = "zkarbie/context-lattice-trade-signal"

    def __init__(
        self,
        model_path: str = None,
        use_onnx: bool = True,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path) if model_path else Path(self.DEFAULT_MODEL)
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

    def classify(self, description: str) -> TradeSignalResult:
        """Classify trade signal from description."""
        if self.onnx_session is not None:
            return self._classify_onnx(description)
        else:
            return self._classify_pytorch(description)

    def _classify_onnx(self, description: str) -> TradeSignalResult:
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

        label = self.id2label.get(pred_id, "hold")
        action = TradeAction(label) if label in [a.value for a in TradeAction] else TradeAction.HOLD

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return TradeSignalResult(
            action=action,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _classify_pytorch(self, description: str) -> TradeSignalResult:
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

        label = self.id2label.get(pred_id, "hold")
        action = TradeAction(label) if label in [a.value for a in TradeAction] else TradeAction.HOLD

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return TradeSignalResult(
            action=action,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def classify_from_prediction(
        self,
        direction: str,
        confidence: float,
        velocity: float,
        rl_agreement: int = 5,
        total_rl_agents: int = 9,
    ) -> TradeSignalResult:
        """
        Classify from FinColl prediction data.

        Args:
            direction: "bullish", "bearish", or "neutral"
            confidence: Prediction confidence (0-1)
            velocity: Price velocity
            rl_agreement: Number of RL agents that agree
            total_rl_agents: Total RL agents

        Returns:
            TradeSignalResult
        """
        description = (
            f"FinColl predicts {direction} with {confidence:.0%} confidence, "
            f"velocity {velocity:.4f}, {rl_agreement} of {total_rl_agents} RL agents agree"
        )
        return self.classify(description)


def test_classifier(model_path: str):
    """Test the classifier with sample descriptions."""
    classifier = TradeSignalClassifier(model_path)

    test_cases = [
        "FinColl predicts bullish with 85% confidence, velocity 0.05 positive, 7 of 9 RL agents agree",
        "FinColl moderately bullish at 60%, velocity 0.02, some RL agents cautious",
        "FinColl neutral with 45% confidence, velocity near zero 0.001",
        "FinColl moderately bearish at 65%, velocity -0.03 negative, some RL agents cautious",
        "Strong sell signal: direction bearish, confidence 88%, momentum negative -0.6",
        "High risk: volatility extreme at 0.15, FinColl confidence only 30%",
    ]

    print("\nTesting trade signal classifier:")
    print("-" * 70)

    for description in test_cases:
        result = classifier.classify(description)
        print(f"Input: {description[:55]}...")
        print(f"  Action: {result.action.value}, Confidence: {result.confidence:.3f}")
        print()


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else TradeSignalClassifier.DEFAULT_MODEL
    test_classifier(model_path)
