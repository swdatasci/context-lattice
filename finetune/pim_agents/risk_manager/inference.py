"""
Fast inference for Risk Manager classification.

Classifies market conditions into risk levels (LOW, MEDIUM, HIGH, EXTREME).

Usage:
    from finetune.pim_agents.risk_manager.inference import RiskClassifier

    classifier = RiskClassifier("models/risk_manager/final")
    result = classifier.classify("High confidence at 85%, low volatility 0.02")
    print(result.risk_level, result.confidence)
"""

import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RiskLevel(Enum):
    """Risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskClassification:
    """Result of risk classification."""
    risk_level: RiskLevel
    confidence: float
    all_scores: dict  # Scores for all risk levels


class RiskClassifier:
    """
    Fine-tuned risk classifier for PIM.

    Supports PyTorch and ONNX backends.
    """

    # Default to Hugging Face Hub model
    DEFAULT_MODEL = "zkarbie/pim-risk-manager"

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

    def classify(self, description: str) -> RiskClassification:
        """Classify risk level from market description."""
        if self.onnx_session is not None:
            return self._classify_onnx(description)
        else:
            return self._classify_pytorch(description)

    def _classify_onnx(self, description: str) -> RiskClassification:
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

        label = self.id2label.get(pred_id, "medium")
        risk_level = RiskLevel(label) if label in [r.value for r in RiskLevel] else RiskLevel.MEDIUM

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return RiskClassification(
            risk_level=risk_level,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _classify_pytorch(self, description: str) -> RiskClassification:
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

        label = self.id2label.get(pred_id, "medium")
        risk_level = RiskLevel(label) if label in [r.value for r in RiskLevel] else RiskLevel.MEDIUM

        all_scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return RiskClassification(
            risk_level=risk_level,
            confidence=confidence,
            all_scores=all_scores,
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def classify_from_metrics(
        self,
        confidence: float,
        volatility: float,
        velocity: float = 0.0,
        acceleration: float = 0.0,
        endpoint_range: float = 0.0,
    ) -> RiskClassification:
        """
        Classify risk from numeric metrics.

        This creates a text description from the metrics and classifies it.
        """
        description = (
            f"Confidence is {confidence:.0%}, volatility is {volatility:.4f}, "
            f"velocity is {velocity:.4f}, acceleration is {acceleration:.4f}, "
            f"endpoint range width is {endpoint_range:.4f}"
        )
        return self.classify(description)


def test_classifier(model_path: str):
    """Test the classifier with sample descriptions."""
    classifier = RiskClassifier(model_path)

    test_cases = [
        "Confidence is 85%, volatility is low at 0.02, velocity is 0.01",
        "Moderate confidence at 55%, volatility elevated at 0.06",
        "High uncertainty with endpoint range 0.08, low confidence 35%",
        "Extreme volatility 0.25, very low confidence 15%",
        "Strong signal with 90% confidence, narrow endpoint range of 0.01",
        "Market too volatile: velocity 0.08, acceleration 0.05",
    ]

    print("\nTesting risk classifier:")
    print("-" * 70)

    for description in test_cases:
        result = classifier.classify(description)
        print(f"Input: {description[:60]}...")
        print(f"  Risk Level: {result.risk_level.value}, Confidence: {result.confidence:.3f}")
        print()


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else RiskClassifier.DEFAULT_MODEL
    test_classifier(model_path)
