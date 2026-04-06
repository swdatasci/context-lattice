"""
Fine-tune DistilBERT for Intent Classification.

Uses HuggingFace Transformers + Datasets for efficient training.
Exports to both PyTorch and ONNX for fast inference.

Requirements:
    pip install transformers datasets torch accelerate onnx onnxruntime

Usage:
    # Generate data first
    python generate_training_data.py --output data --count 200

    # Train
    python train.py --data data --output models/intent_classifier

    # With custom settings
    python train.py --data data --output models --epochs 5 --batch-size 32
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
import numpy as np


# Intent labels (must match generate_training_data.py)
INTENT_LABELS = [
    "CODING",
    "DEBUGGING",
    "REFACTORING",
    "RESEARCH",
    "PLANNING",
    "DOCUMENTATION",
]

LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(INTENT_LABELS)}


class IntentDataset(Dataset):
    """PyTorch dataset for intent classification."""

    def __init__(self, data_path: Path, tokenizer: DistilBertTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example["text"]
        label = LABEL2ID[example["label"]]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """Compute accuracy and per-class metrics."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    accuracy = (preds == labels).mean()

    # Per-class accuracy
    metrics = {"accuracy": accuracy}
    for label_id, label_name in ID2LABEL.items():
        mask = labels == label_id
        if mask.sum() > 0:
            class_acc = (preds[mask] == labels[mask]).mean()
            metrics[f"accuracy_{label_name.lower()}"] = class_acc

    return metrics


def train(
    data_dir: Path,
    output_dir: Path,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    export_onnx: bool = True,
):
    """
    Fine-tune DistilBERT on intent classification.

    Args:
        data_dir: Directory containing train.jsonl and val.jsonl
        output_dir: Directory to save model
        model_name: Base model name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Max sequence length
        export_onnx: Whether to export ONNX model
    """
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print(f"Loading datasets from {data_dir}")
    train_dataset = IntentDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_dataset = IntentDataset(data_dir / "val.jsonl", tokenizer, max_length)

    print(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    # Evaluate
    print("\nEvaluating...")
    metrics = trainer.evaluate()
    print(f"Final metrics: {metrics}")

    # Save model
    final_dir = output_dir / "final"
    print(f"\nSaving model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save label mapping
    with open(final_dir / "label_mapping.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

    # Export to ONNX
    if export_onnx:
        print("\nExporting to ONNX...")
        export_to_onnx(model, tokenizer, final_dir, max_length)

    print("\nTraining complete!")
    return metrics


def export_to_onnx(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    output_dir: Path,
    max_length: int = 128,
):
    """Export model to ONNX format for fast inference."""
    try:
        import onnx
        from transformers.onnx import export

        onnx_path = output_dir / "model.onnx"

        # Create dummy input
        dummy_input = tokenizer(
            "Fix the bug in login.py",
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        # Export
        model.eval()
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )

        print(f"ONNX model saved to {onnx_path}")

        # Verify
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully")

    except ImportError:
        print("ONNX export skipped (install onnx and onnxruntime)")
    except Exception as e:
        print(f"ONNX export failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for intent classification")
    parser.add_argument("--data", type=str, default="data", help="Data directory")
    parser.add_argument("--output", type=str, default="models/intent_classifier", help="Output directory")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")

    args = parser.parse_args()

    data_dir = Path(__file__).parent / args.data
    output_dir = Path(__file__).parent / args.output

    train(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        export_onnx=not args.no_onnx,
    )
