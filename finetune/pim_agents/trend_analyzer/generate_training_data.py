"""
Generate training data for Trend Analyzer classification.

Classifies market conditions into trend types:
- UPTREND: Bullish, prices rising
- DOWNTREND: Bearish, prices falling
- SIDEWAYS: Range-bound, no clear direction

Based on technical indicators and price action.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

# Trend labels
TREND_LABELS = ["uptrend", "downtrend", "sideways"]

# Templates for generating natural language descriptions
TEMPLATES = {
    "uptrend": [
        "Price is {pct:.1%} above 20-day SMA, RSI at {rsi:.0f}, MACD positive",
        "Strong bullish momentum with RSI {rsi:.0f}, price above moving averages",
        "Upward trend: SMA20 > SMA50, momentum indicator at {mom:.2f}",
        "Bullish pattern detected, price up {pct:.1%}, RSI showing strength at {rsi:.0f}",
        "Clear uptrend with positive MACD {macd:.4f}, price above resistance",
        "Higher highs and higher lows, momentum {mom:.2f}, trend strength {trend:.2f}",
        "Price breaking out, RSI {rsi:.0f}, trend indicator positive {trend:.2f}",
        "Strong buying pressure, price {pct:.1%} above SMA, MACD bullish",
    ],
    "downtrend": [
        "Price is {pct:.1%} below 20-day SMA, RSI at {rsi:.0f}, MACD negative",
        "Bearish momentum with RSI {rsi:.0f}, price below moving averages",
        "Downward trend: SMA20 < SMA50, momentum indicator at {mom:.2f}",
        "Bearish pattern detected, price down {pct:.1%}, RSI weak at {rsi:.0f}",
        "Clear downtrend with negative MACD {macd:.4f}, price below support",
        "Lower highs and lower lows, momentum {mom:.2f}, trend strength {trend:.2f}",
        "Price breaking down, RSI {rsi:.0f}, trend indicator negative {trend:.2f}",
        "Strong selling pressure, price {pct:.1%} below SMA, MACD bearish",
    ],
    "sideways": [
        "Price consolidating near 20-day SMA, RSI neutral at {rsi:.0f}",
        "Range-bound trading, no clear direction, RSI {rsi:.0f}",
        "Sideways movement: price oscillating, momentum near zero at {mom:.2f}",
        "Neutral pattern, price flat within {pct:.1%}, RSI {rsi:.0f}",
        "No trend detected, MACD near zero {macd:.4f}, consolidating",
        "Choppy price action, momentum {mom:.2f}, trend strength low {trend:.2f}",
        "Trading range established, RSI {rsi:.0f}, no breakout signals",
        "Consolidation phase, price stable within {pct:.1%} range",
    ],
}


def generate_example(trend_type: str) -> Dict:
    """Generate a single training example for a trend type."""

    # Generate appropriate metric ranges for each trend type
    if trend_type == "uptrend":
        pct_change = random.uniform(0.02, 0.15)  # 2-15% above SMA
        rsi = random.uniform(55, 80)
        macd = random.uniform(0.001, 0.01)
        momentum = random.uniform(0.3, 1.0)
        trend_strength = random.uniform(0.4, 1.0)
    elif trend_type == "downtrend":
        pct_change = -random.uniform(0.02, 0.15)  # 2-15% below SMA
        rsi = random.uniform(20, 45)
        macd = -random.uniform(0.001, 0.01)
        momentum = random.uniform(-1.0, -0.3)
        trend_strength = -random.uniform(0.4, 1.0)
    else:  # sideways
        pct_change = random.uniform(-0.02, 0.02)  # Near SMA
        rsi = random.uniform(40, 60)
        macd = random.uniform(-0.002, 0.002)
        momentum = random.uniform(-0.2, 0.2)
        trend_strength = random.uniform(-0.2, 0.2)

    # Select a random template
    template = random.choice(TEMPLATES[trend_type])

    # Format the template
    text = template.format(
        pct=abs(pct_change),
        rsi=rsi,
        macd=macd,
        mom=momentum,
        trend=trend_strength
    )

    return {
        "text": text,
        "label": trend_type,
        "metadata": {
            "pct_change": round(pct_change, 4),
            "rsi": round(rsi, 2),
            "macd": round(macd, 6),
            "momentum": round(momentum, 4),
            "trend_strength": round(trend_strength, 4)
        }
    }


def generate_dataset(output_dir: Path, examples_per_class: int = 250) -> Dict[str, int]:
    """Generate full training dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate examples for each trend type
    all_examples = []
    for trend_type in TREND_LABELS:
        for _ in range(examples_per_class):
            example = generate_example(trend_type)
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

    counts = {label: sum(1 for e in all_examples if e["label"] == label) for label in TREND_LABELS}

    print(f"Generated {len(train_examples)} training examples")
    print(f"Generated {len(val_examples)} validation examples")
    print(f"Class distribution: {counts}")

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--count", type=int, default=250, help="Examples per class")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(__file__).parent / args.output
    generate_dataset(output_dir, args.count)
