"""
Fine-tune Cross-Encoder for Semantic Reranking.

Reranks search results based on query-document relevance.
Uses sentence-transformers CrossEncoder with DistilRoBERTa.

Usage:
    python generate_training_data.py --output data --count 2000
    python train.py --data data --output models/semantic_reranker
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator


def load_examples(data_path: Path) -> List[InputExample]:
    """Load training examples from JSONL file."""
    examples = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(InputExample(
                texts=[data["query"], data["document"]],
                label=float(data["label"]) / 2.0  # Normalize to 0-1
            ))
    return examples


def load_eval_data(data_path: Path) -> Tuple[List[List[str]], List[float]]:
    """Load evaluation data for CECorrelationEvaluator."""
    sentence_pairs = []
    labels = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            sentence_pairs.append([data["query"], data["document"]])
            labels.append(float(data["label"]) / 2.0)  # Normalize to 0-1
    return sentence_pairs, labels


def train(
    data_dir: Path,
    output_dir: Path,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
):
    """Fine-tune cross-encoder for semantic reranking."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = CrossEncoder(model_name, num_labels=1, max_length=256)

    print(f"Loading datasets from {data_dir}")
    train_examples = load_examples(data_dir / "train.jsonl")
    val_pairs, val_labels = load_eval_data(data_dir / "val.jsonl")

    print(f"Train: {len(train_examples)} examples, Val: {len(val_pairs)} examples")

    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )

    # Create evaluator
    evaluator = CECorrelationEvaluator(val_pairs, val_labels, name="val")

    # Calculate total steps
    total_steps = len(train_dataloader) * epochs

    print(f"\nStarting training for {epochs} epochs ({total_steps} steps)...")

    # Train
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=len(train_dataloader),  # Evaluate every epoch
        output_path=str(output_dir / "final"),
        save_best_model=True,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True,
    )

    print(f"\nModel saved to {output_dir / 'final'}")

    # Final evaluation
    print("\nFinal evaluation:")
    eval_results = evaluator(model)
    # evaluator returns a dict with pearson/spearman correlations
    pearson = float(eval_results.get('eval_val_pearson', eval_results.get('pearson', 0)))
    spearman = float(eval_results.get('eval_val_spearman', eval_results.get('spearman', 0)))
    print(f"Pearson: {pearson:.4f}, Spearman: {spearman:.4f}")

    # Save label info
    with open(output_dir / "final" / "label_info.json", "w") as f:
        json.dump({
            "num_labels": 1,
            "label_meaning": "0=not relevant, 0.5=somewhat relevant, 1=highly relevant",
            "base_model": model_name,
            "epochs": epochs,
            "pearson_correlation": pearson,
            "spearman_correlation": spearman
        }, f, indent=2)

    print("\nTraining complete!")
    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--output", type=str, default="models/semantic_reranker")
    parser.add_argument("--model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)

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
        warmup_steps=args.warmup_steps,
    )
