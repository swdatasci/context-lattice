"""
Fast inference for Trend Analyzer classification.

Classifies market conditions into trend types (UPTREND, DOWNTREND, SIDEWAYS).

Usage:
    from finetune.pim_agents.trend_analyzer.inference import TrendClassifier

    classifier = TrendClassifier("models/trend_analyzer/final")
    result = classifier.classify("Strong bullish momentum with RSI 65")
    print(result.trend, result.confidence)
"""

import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Trend(Enum):
    """Market trend types."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


@dataclass
class TrendClassification:
    """Result of trend classification."""
    trend: Trend
    confidence: float
    all_scores: dict  # Scores for all trend types


class TrendClassifier:
    """
    Fine-tuned trend classifier for PIM.

    Supports PyTorch and ONNX backends.
    """

    # Default to Hugging Face Hub model
    DEFAULT_MODEL = "zkarbie/pim-trend-analyzer"

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

    def classify(self, description: str) -> TrendClassification:
        """Classify trend from market description."""
        if self.onnx_session is not None:
            return self._classify_onnx(description)
        else:
            return self._classify_pytorch(description)

    def _classify_onnx(self, description: str) -> TrendClassification:
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

        label = self.id2label.get(pred_id, "sideways")
        trend = Trend(label) if label in [t.value for t in Trend] else Trend.SIDEWAYS

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return TrendClassification(
            trend=trend,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _classify_pytorch(self, description: str) -> TrendClassification:
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

        label = self.id2label.get(pred_id, "sideways")
        trend = Trend(label) if label in [t.value for t in Trend] else Trend.SIDEWAYS

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return TrendClassification(
            trend=trend,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def classify_from_indicators(
        self,
        rsi: float,
        macd: float,
        pct_from_sma: float,
        momentum: float = 0.0,
    ) -> TrendClassification:
        """
        Classify trend from numeric indicators.

        Args:
            rsi: RSI value (0-100)
            macd: MACD value
            pct_from_sma: Percent distance from 20-day SMA
            momentum: Momentum indicator

        Returns:
            TrendClassification
        """
        description = (
            f"RSI at {rsi:.0f}, MACD {macd:.4f}, "
            f"price {abs(pct_from_sma):.1%} {'above' if pct_from_sma > 0 else 'below'} SMA, "
            f"momentum {momentum:.2f}"
        )
        return self.classify(description)


def test_classifier(model_path: str):
    """Test the classifier with sample descriptions."""
    classifier = TrendClassifier(model_path)

    test_cases = [
        "Price is 8% above 20-day SMA, RSI at 70, MACD positive",
        "Strong bearish momentum with RSI 25, price below moving averages",
        "Range-bound trading, no clear direction, RSI 50",
        "Clear uptrend with positive MACD 0.005, price above resistance",
        "Lower highs and lower lows, momentum -0.6, trend strength -0.7",
        "Price consolidating near 20-day SMA, RSI neutral at 48",
    ]

    print("\nTesting trend classifier:")
    print("-" * 70)

    for description in test_cases:
        result = classifier.classify(description)
        print(f"Input: {description[:55]}...")
        print(f"  Trend: {result.trend.value}, Confidence: {result.confidence:.3f}")
        print()


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else TrendClassifier.DEFAULT_MODEL
    test_classifier(model_path)
