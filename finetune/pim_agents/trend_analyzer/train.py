"""
Fine-tune DistilBERT for Trend Analyzer Classification.

Classifies market conditions into trend types:
- UPTREND, DOWNTREND, SIDEWAYS

Usage:
    python generate_training_data.py --output data --count 250
    python train.py --data data --output models/trend_analyzer
"""

import json
import argparse
from pathlib import Path
from typing import Dict

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


# Trend labels
TREND_LABELS = ["uptrend", "downtrend", "sideways"]
LABEL2ID = {label: i for i, label in enumerate(TREND_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(TREND_LABELS)}


class TrendDataset(Dataset):
    """PyTorch dataset for trend classification."""

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
    """Compute accuracy metrics."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    accuracy = (preds == labels).mean()

    metrics = {"accuracy": accuracy}
    for label_id, label_name in ID2LABEL.items():
        mask = labels == label_id
        if mask.sum() > 0:
            class_acc = (preds[mask] == labels[mask]).mean()
            metrics[f"accuracy_{label_name}"] = class_acc

    return metrics


def train(
    data_dir: Path,
    output_dir: Path,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
    max_length: int = 128,
    export_onnx: bool = True,
):
    """Fine-tune DistilBERT on trend classification."""
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TREND_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print(f"Loading datasets from {data_dir}")
    train_dataset = TrendDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_dataset = TrendDataset(data_dir / "val.jsonl", tokenizer, max_length)

    print(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")

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
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating...")
    metrics = trainer.evaluate()
    print(f"Final metrics: {metrics}")

    final_dir = output_dir / "final"
    print(f"\nSaving model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with open(final_dir / "label_mapping.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

    if export_onnx:
        print("\nExporting to ONNX...")
        export_to_onnx(model, tokenizer, final_dir, max_length)

    print("\nTraining complete!")
    return metrics


def export_to_onnx(model, tokenizer, output_dir: Path, max_length: int = 128):
    """Export model to ONNX."""
    try:
        import onnx

        onnx_path = output_dir / "model.onnx"

        dummy_input = tokenizer(
            "Strong bullish momentum with RSI 65",
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

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

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully")

    except Exception as e:
        print(f"ONNX export failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--output", type=str, default="models/trend_analyzer")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--no-onnx", action="store_true")

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
