"""
ContextLattice CLI: Query-time context optimization.

Usage:
    context-lattice optimize --query "Fix the authentication bug"
    context-lattice optimize --query "..." --budget 20000
"""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from ..core import (
    HierarchyLevel,
    HierarchyConfig,
    ContextNode,
    BudgetCalculator,
    ContextAssembler,
)
from ..retrieval import (
    IntentClassifier,
    PoolSelector,
    VectorRanker,
)

app = typer.Typer(help="ContextLattice: Query-time context optimization")
console = Console()


@app.command()
def optimize(
    query: str = typer.Option(..., "--query", "-q", help="User query to optimize context for"),
    budget: int = typer.Option(20000, "--budget", "-b", help="Total token budget"),
    conversation_tokens: int = typer.Option(0, "--conversation", help="Tokens in conversation"),
    tools_tokens: int = typer.Option(5000, "--tools", help="Tokens in tool definitions"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed statistics"),
):
    """
    Optimize context for a query.

    Example:
        context-lattice optimize --query "Fix the auth bug in login.py" --budget 20000
    """
    console.print(f"\n[bold]Query:[/bold] {query}\n")

    # Step 1: Classify intent
    classifier = IntentClassifier()
    intent_match = classifier.classify(query)
    console.print(
        f"[bold]Intent:[/bold] {intent_match.intent.value} "
        f"(confidence: {intent_match.confidence:.2f})\n"
    )

    # Step 2: Calculate budget
    config = HierarchyConfig()
    budget_calc = BudgetCalculator(config)
    context_budget = budget_calc.calculate(
        conversation_tokens=conversation_tokens,
        tools_tokens=tools_tokens,
        intent=intent_match.intent.value,
    )

    console.print("[bold]Budget Allocation:[/bold]")
    budget_table = Table(show_header=True)
    budget_table.add_column("Level")
    budget_table.add_column("Tokens", justify="right")
    budget_table.add_column("Percentage", justify="right")

    for level in HierarchyLevel:
        tokens = context_budget.per_level[level]
        pct = (tokens / context_budget.total_available * 100) if context_budget.total_available > 0 else 0
        budget_table.add_row(
            level.name,
            str(tokens),
            f"{pct:.1f}%"
        )

    console.print(budget_table)
    console.print()

    # Step 3: Load sample candidates (for MVP demo)
    # In production, this would fetch from semantic search, files, etc.
    console.print("[yellow]Note: Using demo candidates (production will fetch from Qdrant, files, etc.)[/yellow]\n")

    candidates = _create_demo_candidates()

    # Step 4: Assign to pools
    pool_selector = PoolSelector()
    pools = pool_selector.assign_pools(candidates, query)

    console.print("[bold]Pool Assignment:[/bold]")
    pool_table = Table(show_header=True)
    pool_table.add_column("Level")
    pool_table.add_column("Candidates", justify="right")
    pool_table.add_column("Total Tokens", justify="right")

    for level in HierarchyLevel:
        pool_nodes = pools.get(level, [])
        total_tokens = sum(n.tokens for n in pool_nodes)
        pool_table.add_row(
            level.name,
            str(len(pool_nodes)),
            str(total_tokens)
        )

    console.print(pool_table)
    console.print()

    # Step 5: Rank within pools
    console.print("[cyan]Embedding query...[/cyan]")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)

    ranker = VectorRanker(config)
    ranked_pools = ranker.rank_all_pools(pools, query_embedding)

    # Step 6: Select within budget
    selected = {}
    for level in HierarchyLevel:
        ranked = ranked_pools.get(level, [])
        budget_for_level = context_budget.per_level[level]
        selected[level] = ranker.select_within_budget(ranked, budget_for_level)

    console.print("[bold]Selection Results:[/bold]")
    selection_table = Table(show_header=True)
    selection_table.add_column("Level")
    selection_table.add_column("Selected", justify="right")
    selection_table.add_column("Tokens Used", justify="right")
    selection_table.add_column("Budget", justify="right")
    selection_table.add_column("Utilization", justify="right")

    for level in HierarchyLevel:
        nodes = selected.get(level, [])
        used = sum(n.tokens for n in nodes)
        budget_tokens = context_budget.per_level[level]
        util = (used / budget_tokens * 100) if budget_tokens > 0 else 0

        selection_table.add_row(
            level.name,
            str(len(nodes)),
            str(used),
            str(budget_tokens),
            f"{util:.1f}%"
        )

    console.print(selection_table)
    console.print()

    # Step 7: Assemble context
    assembler = ContextAssembler()
    assembled = assembler.assemble(selected, context_budget)

    console.print(f"[bold green]✓ Context optimized:[/bold green] {assembled.total_tokens} tokens")
    console.print(f"[bold]Efficiency:[/bold] {assembled.efficiency:.1%}\n")

    if verbose:
        console.print("[bold]Assembled Context Preview:[/bold]")
        console.print("[dim]" + assembled.text[:500] + "...[/dim]\n")

    return assembled


def _create_demo_candidates() -> list:
    """Create demo candidates for MVP testing."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    candidates = []

    # Structural
    candidates.append(ContextNode(
        id="struct_1",
        content="User preference: Always use TypeScript strict mode. Prefer functional programming patterns.",
        tokens=20,
        level=HierarchyLevel.STRUCTURAL,
        embedding=model.encode("User preference: Always use TypeScript strict mode"),
        metadata={"type": "user_preference"},
    ))

    # Direct (mentioned file)
    candidates.append(ContextNode(
        id="direct_1",
        content="function login(username, password) { /* auth logic */ }",
        tokens=50,
        level=HierarchyLevel.DIRECT,
        embedding=model.encode("function login(username, password)"),
        metadata={"file_path": "src/auth/login.py", "entity_name": "login"},
    ))

    # Implied (same module)
    candidates.append(ContextNode(
        id="implied_1",
        content="function validatePassword(password) { /* validation */ }",
        tokens=30,
        level=HierarchyLevel.IMPLIED,
        embedding=model.encode("function validatePassword(password)"),
        metadata={"file_path": "src/auth/validation.py", "entity_name": "validatePassword"},
    ))

    # Background (architecture)
    candidates.append(ContextNode(
        id="background_1",
        content="# Authentication Architecture\n\nWe use JWT tokens with 15-minute expiry.",
        tokens=40,
        level=HierarchyLevel.BACKGROUND,
        embedding=model.encode("Authentication Architecture JWT tokens"),
        metadata={"file_path": "docs/ARCHITECTURE.md", "type": "documentation"},
    ))

    return candidates


@app.command()
def info():
    """Show ContextLattice configuration and status."""
    console.print("[bold]ContextLattice v0.1.0[/bold]\n")
    console.print("Query-time context optimization for agentic LLM systems\n")

    console.print("[bold]Hierarchy Levels:[/bold]")
    for level in HierarchyLevel:
        console.print(f"  {level.name}: {level.description}")

    console.print()


if __name__ == "__main__":
    app()
