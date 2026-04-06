"""
Generate training data for Trade Signal Classifier.

Classifies FinColl predictions + market context into trading actions:
- STRONG_LONG: High confidence bullish, favorable conditions
- LONG: Moderate bullish signal
- HOLD: Neutral or low confidence
- SHORT: Moderate bearish signal
- STRONG_SHORT: High confidence bearish, favorable conditions
- AVOID: High risk, conflicting signals

Based on FinColl prediction format and Layer 2 RL filtering logic.
"""

import json
import random
from pathlib import Path
from typing import Dict

# Trade action labels
ACTION_LABELS = ["strong_long", "long", "hold", "short", "strong_short", "avoid"]

# Templates for generating natural language descriptions
TEMPLATES = {
    "strong_long": [
        "FinColl predicts bullish with {conf:.0%} confidence, velocity {vel:.4f} positive, 7 of 9 RL agents agree",
        "Strong buy signal: direction bullish, confidence {conf:.0%}, momentum positive {mom:.4f}",
        "All indicators aligned: prediction bullish at {conf:.0%}, acceleration {acc:.4f} positive",
        "FinColl consensus bullish with {conf:.0%} confidence, RSI {rsi:.0f} not overbought",
        "Layer 2 passes: bullish {conf:.0%}, velocity {vel:.4f}, endpoint suggests continuation",
        "High conviction long: FinColl {conf:.0%} bullish, 8/9 RL agents approve",
    ],
    "long": [
        "FinColl moderately bullish at {conf:.0%}, velocity {vel:.4f}, some RL agents cautious",
        "Buy signal: direction bullish, confidence {conf:.0%}, mixed momentum {mom:.4f}",
        "Leaning bullish: prediction {conf:.0%} confidence, 5 of 9 RL agents agree",
        "Moderate long: FinColl bullish {conf:.0%}, velocity slightly positive {vel:.4f}",
        "Layer 2 filter passes with reservations: bullish {conf:.0%}, some volatility",
        "Cautious buy: FinColl {conf:.0%} bullish, RSI {rsi:.0f} neutral",
    ],
    "hold": [
        "FinColl neutral with {conf:.0%} confidence, velocity near zero {vel:.4f}",
        "No clear signal: confidence {conf:.0%}, RL agents split evenly",
        "Sideways prediction: FinColl {conf:.0%}, momentum flat {mom:.4f}",
        "Hold recommended: low conviction {conf:.0%}, conflicting RL votes",
        "Unclear direction: velocity {vel:.4f} near zero, acceleration {acc:.4f} flat",
        "Layer 2 filters out: confidence {conf:.0%} below threshold",
    ],
    "short": [
        "FinColl moderately bearish at {conf:.0%}, velocity {vel:.4f} negative, some RL agents cautious",
        "Sell signal: direction bearish, confidence {conf:.0%}, momentum declining {mom:.4f}",
        "Leaning bearish: prediction {conf:.0%} confidence, 5 of 9 RL agents agree",
        "Moderate short: FinColl bearish {conf:.0%}, velocity slightly negative {vel:.4f}",
        "Layer 2 filter passes with reservations: bearish {conf:.0%}, some support levels",
        "Cautious sell: FinColl {conf:.0%} bearish, RSI {rsi:.0f} neutral",
    ],
    "strong_short": [
        "FinColl predicts bearish with {conf:.0%} confidence, velocity {vel:.4f} negative, 7 of 9 RL agents agree",
        "Strong sell signal: direction bearish, confidence {conf:.0%}, momentum negative {mom:.4f}",
        "All indicators aligned: prediction bearish at {conf:.0%}, acceleration {acc:.4f} negative",
        "FinColl consensus bearish with {conf:.0%} confidence, RSI {rsi:.0f} not oversold",
        "Layer 2 passes: bearish {conf:.0%}, velocity {vel:.4f}, endpoint suggests continuation",
        "High conviction short: FinColl {conf:.0%} bearish, 8/9 RL agents approve",
    ],
    "avoid": [
        "High risk: volatility extreme at {vol:.4f}, FinColl confidence only {conf:.0%}",
        "Conflicting signals: bullish prediction but negative momentum {mom:.4f}, high uncertainty",
        "Avoid trade: RL agents reject signal, risk score {risk:.2f} too high",
        "Do not trade: endpoint range {epr:.4f} too wide, confidence {conf:.0%} insufficient",
        "Risk exceeds threshold: velocity {vel:.4f} extreme, acceleration {acc:.4f} conflicting",
        "Layer 2 rejects: multiple agents veto due to volatility {vol:.4f}",
    ],
}


def generate_example(action: str) -> Dict:
    """Generate a single training example for a trade action."""

    # Generate appropriate metric ranges for each action
    if action == "strong_long":
        confidence = random.uniform(0.75, 0.95)
        velocity = random.uniform(0.02, 0.08)
        acceleration = random.uniform(0.01, 0.04)
        momentum = random.uniform(0.4, 0.8)
        rsi = random.uniform(40, 65)
        volatility = random.uniform(0.01, 0.04)
        risk_score = random.uniform(0.1, 0.3)
        endpoint_range = random.uniform(0.01, 0.03)
    elif action == "long":
        confidence = random.uniform(0.55, 0.75)
        velocity = random.uniform(0.01, 0.04)
        acceleration = random.uniform(0.0, 0.02)
        momentum = random.uniform(0.2, 0.5)
        rsi = random.uniform(45, 60)
        volatility = random.uniform(0.02, 0.06)
        risk_score = random.uniform(0.25, 0.45)
        endpoint_range = random.uniform(0.02, 0.05)
    elif action == "hold":
        confidence = random.uniform(0.35, 0.55)
        velocity = random.uniform(-0.01, 0.01)
        acceleration = random.uniform(-0.01, 0.01)
        momentum = random.uniform(-0.2, 0.2)
        rsi = random.uniform(45, 55)
        volatility = random.uniform(0.02, 0.05)
        risk_score = random.uniform(0.35, 0.55)
        endpoint_range = random.uniform(0.03, 0.06)
    elif action == "short":
        confidence = random.uniform(0.55, 0.75)
        velocity = -random.uniform(0.01, 0.04)
        acceleration = -random.uniform(0.0, 0.02)
        momentum = -random.uniform(0.2, 0.5)
        rsi = random.uniform(40, 55)
        volatility = random.uniform(0.02, 0.06)
        risk_score = random.uniform(0.25, 0.45)
        endpoint_range = random.uniform(0.02, 0.05)
    elif action == "strong_short":
        confidence = random.uniform(0.75, 0.95)
        velocity = -random.uniform(0.02, 0.08)
        acceleration = -random.uniform(0.01, 0.04)
        momentum = -random.uniform(0.4, 0.8)
        rsi = random.uniform(35, 55)
        volatility = random.uniform(0.01, 0.04)
        risk_score = random.uniform(0.1, 0.3)
        endpoint_range = random.uniform(0.01, 0.03)
    else:  # avoid
        confidence = random.uniform(0.25, 0.50)
        velocity = random.uniform(-0.08, 0.08)
        acceleration = random.uniform(-0.05, 0.05)
        momentum = random.uniform(-0.6, 0.6)
        rsi = random.choice([random.uniform(15, 30), random.uniform(70, 85)])
        volatility = random.uniform(0.08, 0.20)
        risk_score = random.uniform(0.6, 0.9)
        endpoint_range = random.uniform(0.08, 0.15)

    # Select a random template
    template = random.choice(TEMPLATES[action])

    # Format the template
    text = template.format(
        conf=confidence,
        vel=velocity,
        acc=acceleration,
        mom=momentum,
        rsi=rsi,
        vol=volatility,
        risk=risk_score,
        epr=endpoint_range
    )

    return {
        "text": text,
        "label": action,
        "metadata": {
            "confidence": round(confidence, 4),
            "velocity": round(velocity, 4),
            "acceleration": round(acceleration, 4),
            "momentum": round(momentum, 4),
            "rsi": round(rsi, 2),
            "volatility": round(volatility, 4),
            "risk_score": round(risk_score, 4),
            "endpoint_range": round(endpoint_range, 4)
        }
    }


def generate_dataset(output_dir: Path, examples_per_class: int = 200) -> Dict[str, int]:
    """Generate full training dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate examples for each action
    all_examples = []
    for action in ACTION_LABELS:
        for _ in range(examples_per_class):
            example = generate_example(action)
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

    counts = {label: sum(1 for e in all_examples if e["label"] == label) for label in ACTION_LABELS}

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
