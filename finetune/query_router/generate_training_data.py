"""
Generate training data for Query Router fine-tuning.

Routes queries to appropriate Caelum services/handlers:
- semantic_search: Knowledge retrieval, finding docs/code
- file_ops: File reading, writing, editing
- trading: PIM, market analysis, trading decisions
- code_analysis: Debugging, refactoring, code review
- database: Schema queries, database operations
- git_ops: Git operations, commits, branches
- general: General questions, conversation

Output: JSONL with {"text": "...", "label": "service_name"} format
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Routing categories
ROUTES = [
    "semantic_search",
    "file_ops",
    "trading",
    "code_analysis",
    "database",
    "git_ops",
    "general",
]

# Templates for each route
TEMPLATES: Dict[str, List[str]] = {
    "semantic_search": [
        "Search for [TOPIC] in the codebase",
        "Find documentation about [TOPIC]",
        "What do we know about [TOPIC]?",
        "Look up [TOPIC] in our knowledge base",
        "Search caelum for [TOPIC]",
        "Find related context for [TOPIC]",
        "What have we done before with [TOPIC]?",
        "Find previous work on [TOPIC]",
        "Search for examples of [TOPIC]",
        "Look for [TOPIC] patterns in the codebase",
        "Find all references to [TOPIC]",
        "What's our approach to [TOPIC]?",
        "Search for [TOPIC] documentation",
        "Find notes about [TOPIC]",
        "Look up how we handle [TOPIC]",
    ],
    "file_ops": [
        "Read [FILE]",
        "Show me [FILE]",
        "Open [FILE]",
        "What's in [FILE]?",
        "Edit [FILE] to [ACTION]",
        "Update [FILE] with [CONTENT]",
        "Create a new file for [PURPOSE]",
        "Write [CONTENT] to [FILE]",
        "Modify [FILE]",
        "Delete [LINE] from [FILE]",
        "Add [CONTENT] to [FILE]",
        "Check the contents of [FILE]",
        "View [FILE]",
        "Look at [FILE]",
        "Change [OLD] to [NEW] in [FILE]",
    ],
    "trading": [
        "What's the current market status?",
        "Analyze [SYMBOL] for trading",
        "Should we buy [SYMBOL]?",
        "Check PIM agent status",
        "What are the current positions?",
        "Run a backtest on [STRATEGY]",
        "What's our portfolio performance?",
        "Analyze the risk for [SYMBOL]",
        "Check trading signals",
        "What does FinColl predict for [SYMBOL]?",
        "Start the trading engine",
        "Stop trading for [SYMBOL]",
        "What's the P/L today?",
        "Check agent recommendations",
        "Run market analysis",
        "What's the trend for [SYMBOL]?",
        "Check Layer 2 RL signals",
        "Analyze sentiment for [SYMBOL]",
    ],
    "code_analysis": [
        "Debug [ERROR] in [FILE]",
        "Why is [COMPONENT] failing?",
        "Fix the bug in [FILE]",
        "Refactor [COMPONENT]",
        "Review the code in [FILE]",
        "What's wrong with [CODE]?",
        "Analyze [FUNCTION] for issues",
        "Check for errors in [FILE]",
        "Improve performance of [COMPONENT]",
        "Clean up [FILE]",
        "Find the bug causing [ERROR]",
        "Optimize [FUNCTION]",
        "Add error handling to [COMPONENT]",
        "Review changes in [FILE]",
        "Check type safety in [FILE]",
    ],
    "database": [
        "What tables store [DATA]?",
        "Show me the schema for [TABLE]",
        "Query [TABLE] for [CONDITION]",
        "What's in the [TABLE] table?",
        "Find database entries for [QUERY]",
        "Check the [TABLE] schema",
        "How is [DATA] stored?",
        "What columns are in [TABLE]?",
        "Run a query on [TABLE]",
        "Search database for [QUERY]",
        "What's the structure of [TABLE]?",
        "List all tables in [DATABASE]",
        "Check PostgreSQL for [DATA]",
        "Find records where [CONDITION]",
        "What databases do we have?",
    ],
    "git_ops": [
        "Commit the changes",
        "Create a new branch for [FEATURE]",
        "What's the git status?",
        "Push to [BRANCH]",
        "Show recent commits",
        "Create a PR for [FEATURE]",
        "Merge [BRANCH] into main",
        "Check git log",
        "Revert the last commit",
        "Stage [FILE] for commit",
        "What branch am I on?",
        "Show git diff",
        "Pull latest changes",
        "Checkout [BRANCH]",
        "Create a commit message for these changes",
    ],
    "general": [
        "Hello",
        "What time is it?",
        "How are you?",
        "Thanks",
        "What can you do?",
        "Help me understand [TOPIC]",
        "Explain [CONCEPT]",
        "What's the status?",
        "Summary of today's work",
        "What should we do next?",
        "Good morning",
        "Let's take a break",
        "What did we accomplish?",
        "Continue where we left off",
        "What's on the todo list?",
    ],
}

# Slot fillers
SLOTS: Dict[str, List[str]] = {
    "[TOPIC]": [
        "authentication", "trading", "database", "API", "agents",
        "fine-tuning", "context-lattice", "PIM", "FinColl", "hooks",
        "semantic search", "caching", "Redis", "Qdrant", "embeddings",
        "cost optimization", "backtesting", "Layer 2", "meta-learner",
    ],
    "[FILE]": [
        "main.py", "server.ts", "config.yaml", "CLAUDE.md", "README.md",
        "pim_service.py", "intent_classifier.py", "settings.json",
        "hierarchy.py", "collector.py", "pre_query.py", "train.py",
    ],
    "[ACTION]": [
        "fix the bug", "add logging", "update the config", "refactor",
        "add error handling", "improve performance", "add tests",
    ],
    "[CONTENT]": [
        "new function", "error handling", "type hints", "documentation",
        "unit tests", "configuration", "imports",
    ],
    "[PURPOSE]": [
        "unit tests", "configuration", "utilities", "models",
        "API endpoints", "data processing", "training",
    ],
    "[LINE]": ["line 42", "the import", "the function", "the class"],
    "[OLD]": ["old_value", "deprecated_function", "hardcoded_value"],
    "[NEW]": ["new_value", "updated_function", "config_value"],
    "[SYMBOL]": [
        "AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMD", "META",
        "AMZN", "SPY", "QQQ", "BTC", "ETH",
    ],
    "[STRATEGY]": [
        "momentum", "mean reversion", "trend following", "breakout",
        "scalping", "swing trading", "position trading",
    ],
    "[COMPONENT]": [
        "authentication", "database", "API", "cache", "queue",
        "scheduler", "logger", "parser", "validator", "handler",
    ],
    "[ERROR]": [
        "TypeError", "ConnectionError", "timeout", "null pointer",
        "authentication failed", "rate limit", "404", "500",
    ],
    "[FUNCTION]": [
        "process_data", "fetch_predictions", "calculate_metrics",
        "validate_input", "handle_request", "parse_response",
    ],
    "[CODE]": [
        "this function", "the loop", "the condition", "the query",
        "the API call", "the handler", "the middleware",
    ],
    "[DATA]": [
        "user data", "trades", "predictions", "metrics", "logs",
        "agent decisions", "portfolio state", "market data",
    ],
    "[TABLE]": [
        "trades", "portfolio_decisions", "agent_performance",
        "metalearner_weights", "users", "sessions", "predictions",
    ],
    "[CONDITION]": [
        "today's trades", "profitable trades", "recent errors",
        "active positions", "pending orders",
    ],
    "[DATABASE]": ["pim_database", "caelum_cluster", "postgres"],
    "[QUERY]": ["recent trades", "agent accuracy", "portfolio value"],
    "[FEATURE]": [
        "authentication", "caching", "fine-tuning", "hooks",
        "cost-optimization", "new-agent", "bugfix",
    ],
    "[BRANCH]": ["main", "develop", "feature/auth", "fix/bug"],
    "[CONCEPT]": [
        "dependency injection", "event sourcing", "microservices",
        "fine-tuning", "embeddings", "transformers",
    ],
}


def expand_template(template: str) -> str:
    """Expand template by filling slots."""
    result = template
    for slot, fillers in SLOTS.items():
        while slot in result:
            result = result.replace(slot, random.choice(fillers), 1)
    return result


def generate_examples(route: str, count: int = 100) -> List[Dict[str, str]]:
    """Generate training examples for a single route."""
    examples = []
    templates = TEMPLATES[route]

    for _ in range(count):
        template = random.choice(templates)
        text = expand_template(template)
        examples.append({"text": text, "label": route})

    return examples


def add_real_examples() -> List[Dict[str, str]]:
    """Add real examples from actual usage."""
    real_examples = [
        # semantic_search
        ("Search for prior knowledge about fine-tuning", "semantic_search"),
        ("What do we know about context-lattice?", "semantic_search"),
        ("Find documentation about hooks", "semantic_search"),
        ("Look up how we handle cost optimization", "semantic_search"),

        # file_ops
        ("Read the CLAUDE.md file", "file_ops"),
        ("Show me the contents of hierarchy.py", "file_ops"),
        ("Edit main.py to add logging", "file_ops"),
        ("Create a new test file", "file_ops"),

        # trading
        ("What's the current portfolio status?", "trading"),
        ("Run a backtest on AAPL", "trading"),
        ("Check the Layer 2 RL predictions", "trading"),
        ("Start the PIM trading engine", "trading"),

        # code_analysis
        ("Debug the authentication error", "code_analysis"),
        ("Fix the bug in the collector", "code_analysis"),
        ("Review the hook implementation", "code_analysis"),
        ("Refactor the intent classifier", "code_analysis"),

        # database
        ("What tables store trading data?", "database"),
        ("Show me the trades table schema", "database"),
        ("Query agent_performance for today", "database"),
        ("Check the PostgreSQL connection", "database"),

        # git_ops
        ("Commit these changes", "git_ops"),
        ("Create a PR for the hook feature", "git_ops"),
        ("What's the git status?", "git_ops"),
        ("Push to the main branch", "git_ops"),

        # general
        ("What's our next step?", "general"),
        ("Summarize what we've done", "general"),
        ("Hello, let's continue", "general"),
        ("Thanks for the help", "general"),
    ]

    return [{"text": t, "label": l} for t, l in real_examples]


def generate_dataset(
    output_path: Path,
    examples_per_route: int = 150,
    train_split: float = 0.8,
) -> Tuple[int, int]:
    """Generate full training dataset."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_examples = []

    for route in ROUTES:
        examples = generate_examples(route, examples_per_route)
        all_examples.extend(examples)
        print(f"Generated {len(examples)} examples for {route}")

    real = add_real_examples()
    all_examples.extend(real)
    print(f"Added {len(real)} real examples")

    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * train_split)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    output_path = Path(__file__).parent / args.output
    train_count, val_count = generate_dataset(output_path, args.count)

    print(f"\nTotal: {train_count + val_count} examples")
    print(f"Routes: {ROUTES}")
