"""
Generate training data for Semantic Reranker.

Creates query-document pairs with relevance labels for training a cross-encoder.
Labels:
- 0: Not relevant
- 1: Somewhat relevant
- 2: Highly relevant

Used for reranking search results from semantic search.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

# Example queries and documents for code/development context
QUERY_TEMPLATES = [
    "How to implement {concept}",
    "Fix bug in {component}",
    "Where is {entity} defined",
    "How does {feature} work",
    "What is the purpose of {component}",
    "Debug {error_type} error",
    "Refactor {component} for better {quality}",
    "Add {feature} to {component}",
]

CONCEPTS = [
    "authentication", "caching", "database connection", "API endpoint",
    "error handling", "logging", "testing", "deployment", "monitoring",
    "rate limiting", "pagination", "search", "file upload", "webhooks"
]

COMPONENTS = [
    "user service", "payment handler", "notification system", "scheduler",
    "data processor", "API gateway", "cache layer", "message queue",
    "authentication middleware", "rate limiter", "session manager"
]

ENTITIES = [
    "UserModel", "PaymentProcessor", "EmailService", "DatabaseConnection",
    "CacheManager", "AuthMiddleware", "RateLimiter", "SessionStore"
]

FEATURES = [
    "real-time updates", "batch processing", "retry logic", "circuit breaker",
    "health checks", "metrics collection", "audit logging", "data validation"
]

ERROR_TYPES = [
    "connection timeout", "authentication failed", "rate limit exceeded",
    "validation error", "null pointer", "memory leak", "deadlock"
]

QUALITIES = [
    "performance", "readability", "maintainability", "testability", "scalability"
]


def generate_query() -> str:
    """Generate a random query."""
    template = random.choice(QUERY_TEMPLATES)
    return template.format(
        concept=random.choice(CONCEPTS),
        component=random.choice(COMPONENTS),
        entity=random.choice(ENTITIES),
        feature=random.choice(FEATURES),
        error_type=random.choice(ERROR_TYPES),
        quality=random.choice(QUALITIES)
    )


def generate_relevant_doc(query: str, relevance: int) -> str:
    """Generate a document with specified relevance to the query."""

    # Extract key terms from query
    query_lower = query.lower()

    if relevance == 2:  # Highly relevant
        # Document directly addresses the query
        if "implement" in query_lower:
            return f"Implementation guide for {query.split('implement ')[1] if 'implement ' in query_lower else 'the feature'}. " \
                   f"Step-by-step instructions with code examples. Covers setup, configuration, and best practices."
        elif "fix" in query_lower or "bug" in query_lower:
            return f"Troubleshooting guide for common issues. " \
                   f"Root cause analysis and solutions. Includes debugging steps and workarounds."
        elif "where" in query_lower or "defined" in query_lower:
            return f"Source code location and module structure. " \
                   f"Class definitions and interface documentation. File paths and import statements."
        elif "how does" in query_lower or "work" in query_lower:
            return f"Architecture overview and data flow explanation. " \
                   f"Component interactions and processing pipeline. Internal mechanisms detailed."
        elif "purpose" in query_lower:
            return f"Design rationale and use cases. " \
                   f"Why this component exists and what problems it solves. Integration points."
        elif "debug" in query_lower:
            return f"Debugging strategies and common error patterns. " \
                   f"Log analysis and stack trace interpretation. Diagnostic tools and techniques."
        elif "refactor" in query_lower:
            return f"Refactoring patterns and code improvement strategies. " \
                   f"Before/after examples with measurable improvements. Migration guide."
        elif "add" in query_lower:
            return f"Feature implementation guide with API extensions. " \
                   f"Configuration options and customization points. Testing requirements."
        else:
            return f"Comprehensive documentation covering the topic in detail. " \
                   f"Code examples, best practices, and common pitfalls to avoid."

    elif relevance == 1:  # Somewhat relevant
        # Document is related but not directly addressing the query
        topics = ["architecture", "configuration", "deployment", "testing", "monitoring"]
        return f"General documentation about {random.choice(topics)}. " \
               f"Contains some related information but focuses on broader topics. " \
               f"May require additional context to be fully useful."

    else:  # Not relevant (relevance == 0)
        # Document is about something completely different
        unrelated = [
            "Company holiday schedule and PTO policy. HR contact information.",
            "Meeting notes from quarterly planning. Budget allocations.",
            "Marketing campaign results and social media metrics.",
            "Office supply inventory and ordering procedures.",
            "Employee onboarding checklist and training materials.",
        ]
        return random.choice(unrelated)


def generate_example() -> Dict:
    """Generate a single training example (query, document, relevance)."""
    query = generate_query()
    relevance = random.choices([0, 1, 2], weights=[0.3, 0.3, 0.4])[0]
    document = generate_relevant_doc(query, relevance)

    return {
        "query": query,
        "document": document,
        "label": relevance
    }


def generate_dataset(output_dir: Path, num_examples: int = 2000) -> Dict[str, int]:
    """Generate full training dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate examples
    all_examples = [generate_example() for _ in range(num_examples)]

    # Shuffle
    random.shuffle(all_examples)

    # Split 80/20 train/val
    split_idx = int(len(all_examples) * 0.8)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Write train.jsonl
    with open(output_dir / "train.jsonl", "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")

    # Write val.jsonl
    with open(output_dir / "val.jsonl", "w") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")

    counts = {str(i): sum(1 for e in all_examples if e["label"] == i) for i in range(3)}

    print(f"Generated {len(train_examples)} training examples")
    print(f"Generated {len(val_examples)} validation examples")
    print(f"Label distribution: {counts}")

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--count", type=int, default=2000, help="Total examples")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(__file__).parent / args.output
    generate_dataset(output_dir, args.count)
