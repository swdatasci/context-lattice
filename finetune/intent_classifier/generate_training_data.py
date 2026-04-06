"""
Generate training data for Intent Classifier fine-tuning.

Expands rule-based patterns into diverse training examples using:
1. Template expansion with variations
2. Synonym substitution
3. Real examples from conversation history (if available)

Output: JSONL file with {"text": "...", "label": "DEBUGGING"} format
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Intent categories matching context-lattice
INTENTS = [
    "CODING",
    "DEBUGGING",
    "REFACTORING",
    "RESEARCH",
    "PLANNING",
    "DOCUMENTATION",
]

# Template-based training examples
# Each intent has multiple templates with [SLOT] placeholders
TEMPLATES: Dict[str, List[str]] = {
    "DEBUGGING": [
        "Fix the [ERROR] in [FILE]",
        "Why is [COMPONENT] [FAILING]?",
        "Debug the [ERROR] when [ACTION]",
        "[COMPONENT] is [FAILING], can you help?",
        "Getting [ERROR] error in [FILE]",
        "The [COMPONENT] doesn't work",
        "Can't [ACTION] because of [ERROR]",
        "Investigate why [COMPONENT] is [FAILING]",
        "Troubleshoot the [ERROR] issue",
        "Stack trace shows [ERROR] in [FILE]",
        "Exception [ERROR] thrown by [COMPONENT]",
        "Tests failing with [ERROR]",
        "Diagnose the [COMPONENT] crash",
        "[COMPONENT] throws [ERROR] on [ACTION]",
        "Resolve the [ERROR] bug",
    ],
    "RESEARCH": [
        "How does [COMPONENT] work?",
        "What is [CONCEPT]?",
        "Where is [COMPONENT] defined?",
        "Explain [CONCEPT] to me",
        "Show me how [COMPONENT] handles [ACTION]",
        "What does [COMPONENT] do?",
        "How do I [ACTION] with [COMPONENT]?",
        "Tell me about [CONCEPT]",
        "Describe the [COMPONENT] architecture",
        "What's the purpose of [COMPONENT]?",
        "Find where [CONCEPT] is implemented",
        "Understand the [COMPONENT] flow",
        "Learn about [CONCEPT]",
        "Explore the [COMPONENT] code",
        "What is the [CONCEPT] pattern used here?",
    ],
    "CODING": [
        "Implement [FEATURE] in [FILE]",
        "Add [FEATURE] to [COMPONENT]",
        "Create a new [COMPONENT] for [FEATURE]",
        "Write a [COMPONENT] that [ACTION]",
        "Build [FEATURE] functionality",
        "Make a [COMPONENT] to handle [ACTION]",
        "Develop [FEATURE] for the [COMPONENT]",
        "Code a [COMPONENT] that can [ACTION]",
        "Add a new [FEATURE] endpoint",
        "Implement the [CONCEPT] logic",
        "Create [COMPONENT] class",
        "Write function to [ACTION]",
        "Build integration with [COMPONENT]",
        "Add [FEATURE] support",
        "Implement [ACTION] handler",
    ],
    "REFACTORING": [
        "Refactor [COMPONENT] for better [QUALITY]",
        "Clean up the [COMPONENT] code",
        "Restructure [FILE] to improve [QUALITY]",
        "Extract [COMPONENT] into separate [TARGET]",
        "Improve [COMPONENT] [QUALITY]",
        "Optimize [COMPONENT] performance",
        "Split [COMPONENT] into smaller [TARGET]s",
        "Consolidate [COMPONENT] logic",
        "Deduplicate code in [FILE]",
        "Polish the [COMPONENT] implementation",
        "Reorganize [FILE] structure",
        "Simplify [COMPONENT] logic",
        "Merge [COMPONENT] modules",
        "Enhance [COMPONENT] [QUALITY]",
        "Clean up [FILE]",
    ],
    "PLANNING": [
        "Plan the [FEATURE] implementation",
        "Design [COMPONENT] architecture",
        "What's the best approach for [FEATURE]?",
        "Should we [DECISION]?",
        "Evaluate options for [FEATURE]",
        "Strategy for implementing [FEATURE]",
        "Compare [OPTION1] vs [OPTION2]",
        "Decide between [OPTION1] and [OPTION2]",
        "What if we [DECISION]?",
        "Consider the [CONCEPT] approach",
        "Roadmap for [FEATURE]",
        "Architecture decision for [COMPONENT]",
        "How should we structure [FEATURE]?",
        "Design the [COMPONENT] system",
        "Plan migration to [TARGET]",
    ],
    "DOCUMENTATION": [
        "Document [COMPONENT]",
        "Write docs for [FEATURE]",
        "Add docstrings to [FILE]",
        "Create README for [COMPONENT]",
        "Update documentation for [FEATURE]",
        "Write API docs for [COMPONENT]",
        "Add comments to [FILE]",
        "Document the [CONCEPT] pattern",
        "Create user guide for [FEATURE]",
        "Write tutorial for [COMPONENT]",
        "Add type hints to [FILE]",
        "Document [ACTION] workflow",
        "Create changelog for [FEATURE]",
        "Write inline documentation",
        "Update README with [FEATURE]",
    ],
}

# Slot fillers for template expansion
SLOTS: Dict[str, List[str]] = {
    "[ERROR]": [
        "TypeError", "AttributeError", "KeyError", "ValueError", "ImportError",
        "ConnectionError", "timeout", "null pointer", "undefined", "NaN",
        "authentication", "permission", "404", "500", "memory", "overflow",
    ],
    "[FILE]": [
        "main.py", "server.ts", "index.js", "config.yaml", "utils.py",
        "auth.py", "database.py", "api.ts", "handler.py", "service.ts",
        "login.py", "user.py", "payment.ts", "order.py", "checkout.js",
    ],
    "[COMPONENT]": [
        "authentication", "database", "API", "server", "client",
        "cache", "queue", "scheduler", "logger", "parser",
        "validator", "serializer", "handler", "middleware", "router",
        "service", "controller", "model", "view", "component",
    ],
    "[FAILING]": [
        "not working", "broken", "failing", "crashing", "timing out",
        "returning null", "throwing errors", "slow", "unresponsive", "stuck",
    ],
    "[ACTION]": [
        "login", "submit", "fetch", "save", "delete", "update",
        "process", "validate", "parse", "serialize", "authenticate",
        "connect", "disconnect", "initialize", "shutdown", "restart",
    ],
    "[FEATURE]": [
        "user authentication", "file upload", "search", "pagination",
        "caching", "rate limiting", "logging", "notifications", "analytics",
        "export", "import", "backup", "sync", "webhooks", "API versioning",
    ],
    "[CONCEPT]": [
        "dependency injection", "event sourcing", "CQRS", "microservices",
        "REST API", "GraphQL", "WebSocket", "authentication flow", "middleware",
        "caching strategy", "database indexing", "connection pooling",
    ],
    "[QUALITY]": [
        "readability", "maintainability", "performance", "testability",
        "modularity", "reusability", "clarity", "simplicity",
    ],
    "[TARGET]": [
        "function", "class", "module", "file", "service", "component",
    ],
    "[DECISION]": [
        "use microservices", "add caching", "switch to TypeScript",
        "add tests first", "refactor before adding features",
        "use a database", "add authentication", "deploy to cloud",
    ],
    "[OPTION1]": ["Redis", "PostgreSQL", "REST", "monolith", "Python"],
    "[OPTION2]": ["Memcached", "MongoDB", "GraphQL", "microservices", "Go"],
}


def expand_template(template: str) -> str:
    """Expand a template by filling slots with random values."""
    result = template
    for slot, fillers in SLOTS.items():
        while slot in result:
            result = result.replace(slot, random.choice(fillers), 1)
    return result


def generate_examples(intent: str, count: int = 100) -> List[Dict[str, str]]:
    """Generate training examples for a single intent."""
    examples = []
    templates = TEMPLATES[intent]

    for _ in range(count):
        template = random.choice(templates)
        text = expand_template(template)
        examples.append({"text": text, "label": intent})

    return examples


def add_real_examples() -> List[Dict[str, str]]:
    """Add real examples from conversation history if available."""
    examples = []

    # Hard-coded real examples for better coverage
    real_examples = [
        # DEBUGGING
        ("Fix the authentication bug in login.py", "DEBUGGING"),
        ("Why is the server crashing on startup?", "DEBUGGING"),
        ("Debug memory leak in the worker process", "DEBUGGING"),
        ("The API returns 500 error intermittently", "DEBUGGING"),
        ("Tests are failing in CI but pass locally", "DEBUGGING"),

        # RESEARCH
        ("How does the context-lattice hierarchy work?", "RESEARCH"),
        ("What is the purpose of the pool selector?", "RESEARCH"),
        ("Where are the agent weights stored?", "RESEARCH"),
        ("Explain the intent classification system", "RESEARCH"),
        ("Show me how embeddings are computed", "RESEARCH"),

        # CODING
        ("Add a pre-query hook for Claude Code", "CODING"),
        ("Implement cost-aware escalation levels", "CODING"),
        ("Create a new source for todo items", "CODING"),
        ("Write a function to calculate token budget", "CODING"),
        ("Build the feedback tracking system", "CODING"),

        # REFACTORING
        ("Refactor the collector for better error handling", "REFACTORING"),
        ("Clean up the CLI code", "REFACTORING"),
        ("Extract common logic into a base class", "REFACTORING"),
        ("Simplify the budget calculation", "REFACTORING"),
        ("Optimize the vector ranking performance", "REFACTORING"),

        # PLANNING
        ("Plan the Phase 3 implementation", "PLANNING"),
        ("Design the cost escalation strategy", "PLANNING"),
        ("Should we use Redis or file-based caching?", "PLANNING"),
        ("What's the best approach for hook integration?", "PLANNING"),
        ("Evaluate fine-tuning vs rule-based classification", "PLANNING"),

        # DOCUMENTATION
        ("Document the hook installation process", "DOCUMENTATION"),
        ("Write README for the CLI commands", "DOCUMENTATION"),
        ("Add docstrings to the assembler module", "DOCUMENTATION"),
        ("Create API documentation for sources", "DOCUMENTATION"),
        ("Update CLAUDE.md with new features", "DOCUMENTATION"),
    ]

    for text, label in real_examples:
        examples.append({"text": text, "label": label})

    return examples


def generate_dataset(
    output_path: Path,
    examples_per_intent: int = 200,
    train_split: float = 0.8,
) -> Tuple[int, int]:
    """
    Generate full training dataset.

    Args:
        output_path: Directory to save train.jsonl and val.jsonl
        examples_per_intent: Number of synthetic examples per intent
        train_split: Fraction for training (rest is validation)

    Returns:
        Tuple of (train_count, val_count)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_examples = []

    # Generate synthetic examples for each intent
    for intent in INTENTS:
        examples = generate_examples(intent, examples_per_intent)
        all_examples.extend(examples)
        print(f"Generated {len(examples)} examples for {intent}")

    # Add real examples
    real = add_real_examples()
    all_examples.extend(real)
    print(f"Added {len(real)} real examples")

    # Shuffle
    random.shuffle(all_examples)

    # Split
    split_idx = int(len(all_examples) * train_split)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Write files
    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    with open(train_path, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")

    with open(val_path, "w") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nSaved {len(train_examples)} training examples to {train_path}")
    print(f"Saved {len(val_examples)} validation examples to {val_path}")

    return len(train_examples), len(val_examples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate intent classifier training data")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--count", type=int, default=200, help="Examples per intent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    output_path = Path(__file__).parent / args.output
    train_count, val_count = generate_dataset(output_path, args.count)

    print(f"\nTotal: {train_count + val_count} examples")
    print(f"Labels: {INTENTS}")
