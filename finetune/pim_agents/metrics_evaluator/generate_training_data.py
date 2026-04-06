"""
Generate training data for Metrics Evaluator classification.

Classifies agent performance signals:
- STRONG_BUY: Multiple agents agree on bullish signal with high confidence
- BUY: Moderate bullish consensus
- NEUTRAL: Mixed signals or low confidence
- SELL: Moderate bearish consensus
- STRONG_SELL: Multiple agents agree on bearish signal with high confidence

Based on committee voting patterns from PIM.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

# Signal labels
SIGNAL_LABELS = ["strong_buy", "buy", "neutral", "sell", "strong_sell"]

# Agent names (matching PIM's Layer 1 agents)
AGENTS = [
    "MomentumAgent", "TechnicalAgent", "SentimentAgent",
    "RiskAgent", "TrendAgent", "MacroAgent", "VolumeAgent"
]

# Templates for generating natural language descriptions
TEMPLATES = {
    "strong_buy": [
        "Committee vote: {votes_for}/{total} agents recommend LONG, average confidence {conf:.0%}",
        "Strong consensus: {agents} all bullish with high confidence {conf:.0%}",
        "Agreement {agree:.0%}: Most agents recommend BUY, weighted confidence {conf:.0%}",
        "Bullish signals from {votes_for} agents: {agents}, consensus confidence {conf:.0%}",
        "{votes_for} agents recommend LONG with {conf:.0%} confidence, strong agreement",
        "Committee strongly bullish: {agree:.0%} agreement, confidence {conf:.0%}",
    ],
    "buy": [
        "Moderate bullish: {votes_for}/{total} agents recommend LONG, confidence {conf:.0%}",
        "Mild consensus: {agents} lean bullish, confidence {conf:.0%}",
        "{votes_for} agents bullish, {votes_against} bearish, net positive with {conf:.0%} confidence",
        "Slight buy signal: agreement {agree:.0%}, weighted confidence {conf:.0%}",
        "Committee leans LONG: {votes_for} agents agree at {conf:.0%} confidence",
    ],
    "neutral": [
        "Mixed signals: {votes_for} LONG, {votes_against} SHORT, {neutral} HOLD at {conf:.0%}",
        "No consensus: agents split evenly, low confidence {conf:.0%}",
        "Conflicting views from {agents}, average confidence only {conf:.0%}",
        "Committee undecided: {agree:.0%} agreement, recommend HOLD",
        "Low conviction: {neutral}/{total} agents recommend HOLD",
        "Signals unclear: confidence {conf:.0%} below threshold",
    ],
    "sell": [
        "Moderate bearish: {votes_against}/{total} agents recommend SHORT, confidence {conf:.0%}",
        "Mild consensus: {agents} lean bearish, confidence {conf:.0%}",
        "{votes_against} agents bearish, {votes_for} bullish, net negative with {conf:.0%} confidence",
        "Slight sell signal: agreement {agree:.0%}, weighted confidence {conf:.0%}",
        "Committee leans SHORT: {votes_against} agents agree at {conf:.0%} confidence",
    ],
    "strong_sell": [
        "Committee vote: {votes_against}/{total} agents recommend SHORT, average confidence {conf:.0%}",
        "Strong consensus: {agents} all bearish with high confidence {conf:.0%}",
        "Agreement {agree:.0%}: Most agents recommend SELL, weighted confidence {conf:.0%}",
        "Bearish signals from {votes_against} agents: {agents}, consensus confidence {conf:.0%}",
        "{votes_against} agents recommend SHORT with {conf:.0%} confidence, strong agreement",
        "Committee strongly bearish: {agree:.0%} agreement, confidence {conf:.0%}",
    ],
}


def generate_example(signal_type: str) -> Dict:
    """Generate a single training example for a signal type."""

    total = len(AGENTS)

    # Generate appropriate vote distributions for each signal type
    if signal_type == "strong_buy":
        votes_for = random.randint(5, 7)  # 5-7 bullish
        votes_against = random.randint(0, 1)
        neutral = total - votes_for - votes_against
        confidence = random.uniform(0.75, 0.95)
        agreement = random.uniform(0.75, 0.95)
    elif signal_type == "buy":
        votes_for = random.randint(4, 5)
        votes_against = random.randint(1, 2)
        neutral = total - votes_for - votes_against
        confidence = random.uniform(0.55, 0.75)
        agreement = random.uniform(0.55, 0.75)
    elif signal_type == "neutral":
        votes_for = random.randint(2, 3)
        votes_against = random.randint(2, 3)
        neutral = total - votes_for - votes_against
        confidence = random.uniform(0.35, 0.55)
        agreement = random.uniform(0.35, 0.55)
    elif signal_type == "sell":
        votes_against = random.randint(4, 5)
        votes_for = random.randint(1, 2)
        neutral = total - votes_for - votes_against
        confidence = random.uniform(0.55, 0.75)
        agreement = random.uniform(0.55, 0.75)
    else:  # strong_sell
        votes_against = random.randint(5, 7)
        votes_for = random.randint(0, 1)
        neutral = total - votes_for - votes_against
        confidence = random.uniform(0.75, 0.95)
        agreement = random.uniform(0.75, 0.95)

    # Ensure non-negative
    neutral = max(0, neutral)

    # Select random agents for the description
    sample_agents = random.sample(AGENTS, min(3, max(votes_for, votes_against)))
    agents_str = ", ".join(sample_agents)

    # Select a random template
    template = random.choice(TEMPLATES[signal_type])

    # Format the template
    text = template.format(
        votes_for=votes_for,
        votes_against=votes_against,
        neutral=neutral,
        total=total,
        conf=confidence,
        agree=agreement,
        agents=agents_str
    )

    return {
        "text": text,
        "label": signal_type,
        "metadata": {
            "votes_for": votes_for,
            "votes_against": votes_against,
            "neutral": neutral,
            "confidence": round(confidence, 4),
            "agreement": round(agreement, 4)
        }
    }


def generate_dataset(output_dir: Path, examples_per_class: int = 200) -> Dict[str, int]:
    """Generate full training dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate examples for each signal type
    all_examples = []
    for signal_type in SIGNAL_LABELS:
        for _ in range(examples_per_class):
            example = generate_example(signal_type)
            all_examples.append(example)

    # Shuffle
    random.shuffle(all_examples)

    # Split 80/20 train/val
    split_idx = int(len(all_examples) * 0.8)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Write train.jsonl
    with open(output_dir / "train.jsonl", "w") as f:
        for example in train_examples:
            f.write(json.dumps({"text": example["text"], "label": example["label"]}) + "\n")

    # Write val.jsonl
    with open(output_dir / "val.jsonl", "w") as f:
        for example in val_examples:
            f.write(json.dumps({"text": example["text"], "label": example["label"]}) + "\n")

    # Write full.jsonl with metadata
    with open(output_dir / "full.jsonl", "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    counts = {label: sum(1 for e in all_examples if e["label"] == label) for label in SIGNAL_LABELS}

    print(f"Generated {len(train_examples)} training examples")
    print(f"Generated {len(val_examples)} validation examples")
    print(f"Class distribution: {counts}")

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--count", type=int, default=200, help="Examples per class")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(__file__).parent / args.output
    generate_dataset(output_dir, args.count)
