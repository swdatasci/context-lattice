"""
Generate training data for Risk Manager classification.

Classifies market conditions into risk levels:
- LOW: Safe to trade with normal position sizing
- MEDIUM: Proceed with reduced position size
- HIGH: Proceed with caution, tight stops
- EXTREME: Do not trade, excessive risk

Based on rule-based logic from PIM's RiskEquitiesAgent.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

# Risk levels
RISK_LEVELS = ["low", "medium", "high", "extreme"]

# Templates for generating natural language descriptions
TEMPLATES = {
    "low": [
        "Confidence is {conf:.0%}, volatility is low at {vol:.4f}, velocity is {vel:.4f}",
        "Strong signal with {conf:.0%} confidence, narrow endpoint range of {epr:.4f}",
        "Low risk environment: volatility {vol:.4f}, acceleration {acc:.4f}",
        "Favorable conditions with confidence {conf:.0%} and moderate velocity {vel:.4f}",
        "Clear signal detected, confidence {conf:.0%}, endpoint range width {epr:.4f}",
        "Market conditions favorable: vol={vol:.4f}, confidence={conf:.0%}",
        "Risk score indicates safe entry: confidence {conf:.0%}, volatility estimate {vol:.4f}",
        "Stable market with low volatility {vol:.4f}, high confidence {conf:.0%}",
    ],
    "medium": [
        "Moderate confidence at {conf:.0%}, volatility elevated at {vol:.4f}",
        "Some uncertainty with endpoint range {epr:.4f}, velocity {vel:.4f}",
        "Medium risk: confidence {conf:.0%}, volatility estimate {vol:.4f}",
        "Proceed with caution, acceleration at {acc:.4f}, confidence {conf:.0%}",
        "Mixed signals: confidence {conf:.0%} but velocity magnitude {vel:.4f}",
        "Risk score moderate: volatility {vol:.4f}, endpoint range {epr:.4f}",
        "Borderline conditions: confidence {conf:.0%}, recommend reduced size",
        "Market showing some volatility {vol:.4f} with confidence {conf:.0%}",
    ],
    "high": [
        "High uncertainty with endpoint range {epr:.4f}, low confidence {conf:.0%}",
        "Elevated volatility {vol:.4f}, extreme velocity magnitude {vel:.4f}",
        "Risk score elevated: confidence only {conf:.0%}, volatility {vol:.4f}",
        "Dangerous conditions: acceleration {acc:.4f}, wide endpoint range {epr:.4f}",
        "High risk environment: confidence {conf:.0%}, velocity {vel:.4f}",
        "Volatility estimate high at {vol:.4f}, recommend tight stops",
        "Uncertain market: confidence {conf:.0%}, endpoint range width {epr:.4f}",
        "Elevated risk detected: vol={vol:.4f}, velocity={vel:.4f}",
    ],
    "extreme": [
        "Extreme volatility {vol:.4f}, very low confidence {conf:.0%}",
        "Do not trade: risk score above threshold, volatility {vol:.4f}",
        "Extreme uncertainty with endpoint range {epr:.4f}, confidence only {conf:.0%}",
        "Market too volatile: velocity {vel:.4f}, acceleration {acc:.4f}",
        "Excessive risk: volatility estimate {vol:.4f}, confidence {conf:.0%}",
        "Risk exceeds threshold: do not enter, vol={vol:.4f}",
        "Extreme conditions detected: wide endpoint range {epr:.4f}, low confidence {conf:.0%}",
        "Market exhaustion risk: extreme velocity {vel:.4f}, high volatility {vol:.4f}",
    ],
}


def calculate_risk_level(confidence: float, volatility: float, velocity: float,
                         acceleration: float, endpoint_range: float) -> str:
    """
    Calculate risk level based on PIM's rule-based logic.

    Risk score = (1 - confidence) * 0.5 + min(volatility, 1.0) * 0.5
    - risk_score > 0.7 → EXTREME
    - risk_score > 0.5 → HIGH
    - risk_score > 0.3 → MEDIUM
    - else → LOW

    Also considers velocity magnitude for exhaustion risk.
    """
    # Calculate risk score
    risk_score = (1.0 - confidence) * 0.5 + min(volatility, 1.0) * 0.5

    # Extreme velocity adds to risk
    if abs(velocity) > 0.05:
        risk_score += 0.15

    # Classify risk level
    if risk_score > 0.7:
        return "extreme"
    elif risk_score > 0.5:
        return "high"
    elif risk_score > 0.3:
        return "medium"
    else:
        return "low"


def generate_example(risk_level: str) -> Dict:
    """Generate a single training example for a risk level."""

    # Generate appropriate metric ranges for each risk level
    if risk_level == "low":
        confidence = random.uniform(0.65, 0.95)
        volatility = random.uniform(0.01, 0.04)
        velocity = random.uniform(-0.03, 0.03)
        acceleration = random.uniform(-0.02, 0.02)
        endpoint_range = random.uniform(0.005, 0.02)
    elif risk_level == "medium":
        confidence = random.uniform(0.45, 0.65)
        volatility = random.uniform(0.03, 0.08)
        velocity = random.uniform(-0.04, 0.04)
        acceleration = random.uniform(-0.03, 0.03)
        endpoint_range = random.uniform(0.02, 0.05)
    elif risk_level == "high":
        confidence = random.uniform(0.30, 0.50)
        volatility = random.uniform(0.06, 0.15)
        velocity = random.uniform(-0.06, 0.06)
        acceleration = random.uniform(-0.04, 0.04)
        endpoint_range = random.uniform(0.04, 0.10)
    else:  # extreme
        confidence = random.uniform(0.10, 0.35)
        volatility = random.uniform(0.10, 0.30)
        velocity = random.uniform(-0.10, 0.10)
        acceleration = random.uniform(-0.06, 0.06)
        endpoint_range = random.uniform(0.08, 0.20)

    # Verify the calculated risk level matches (with some overlap for realism)
    calc_risk = calculate_risk_level(confidence, volatility, velocity, acceleration, endpoint_range)

    # Select a random template
    template = random.choice(TEMPLATES[risk_level])

    # Format the template
    text = template.format(
        conf=confidence,
        vol=volatility,
        vel=velocity,
        acc=acceleration,
        epr=endpoint_range
    )

    return {
        "text": text,
        "label": risk_level,
        "metadata": {
            "confidence": round(confidence, 4),
            "volatility": round(volatility, 4),
            "velocity": round(velocity, 4),
            "acceleration": round(acceleration, 4),
            "endpoint_range": round(endpoint_range, 4),
            "calculated_risk": calc_risk
        }
    }


def generate_dataset(output_dir: Path, examples_per_class: int = 200) -> Dict[str, int]:
    """Generate full training dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate examples for each risk level
    all_examples = []
    for risk_level in RISK_LEVELS:
        for _ in range(examples_per_class):
            example = generate_example(risk_level)
            all_examples.append(example)

    # Shuffle
    random.shuffle(all_examples)

    # Split 80/20 train/val
    split_idx = int(len(all_examples) * 0.8)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Write train.jsonl (without metadata for cleaner training)
    with open(output_dir / "train.jsonl", "w") as f:
        for example in train_examples:
            f.write(json.dumps({"text": example["text"], "label": example["label"]}) + "\n")

    # Write val.jsonl
    with open(output_dir / "val.jsonl", "w") as f:
        for example in val_examples:
            f.write(json.dumps({"text": example["text"], "label": example["label"]}) + "\n")

    # Write full.jsonl with metadata for analysis
    with open(output_dir / "full.jsonl", "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    counts = {level: sum(1 for e in all_examples if e["label"] == level) for level in RISK_LEVELS}

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
