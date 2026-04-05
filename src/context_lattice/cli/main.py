"""
ContextLattice CLI: Query-time context optimization.

Usage:
    context-lattice optimize --query "Fix the authentication bug"
    context-lattice optimize --query "..." --project-root /path/to/project
    context-lattice optimize --query "..." --sources semantic,file --no-cache
"""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional, List
from pathlib import Path
import yaml

from ..core import (
    HierarchyLevel,
    HierarchyConfig,
    BudgetCalculator,
    ContextAssembler,
)
from ..retrieval import (
    IntentClassifier,
    PoolSelector,
    VectorRanker,
)
from ..sources import MultiSourceCollector
from ..feedback import FeedbackTracker
from ..hooks import PreQueryHook

app = typer.Typer(help="ContextLattice: Query-time context optimization")
console = Console()


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default config location
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        console.print(f"[yellow]Warning: Config file not found at {config_path}, using defaults[/yellow]")
        return {}


@app.command()
def optimize(
    query: str = typer.Option(..., "--query", "-q", help="User query to optimize context for"),
    budget: int = typer.Option(20000, "--budget", "-b", help="Total token budget"),
    conversation_tokens: int = typer.Option(0, "--conversation", help="Tokens in conversation"),
    tools_tokens: int = typer.Option(5000, "--tools", help="Tokens in tool definitions"),
    project_root: Optional[Path] = typer.Option(None, "--project-root", "-p", help="Project root directory"),
    current_file: Optional[str] = typer.Option(None, "--current-file", "-f", help="Currently open file"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Comma-separated sources (semantic,file)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    track_feedback: bool = typer.Option(True, "--track-feedback", help="Enable feedback tracking"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed statistics"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Config file path"),
):
    """
    Optimize context for a query using real sources.

    Example:
        context-lattice optimize --query "Fix the auth bug in login.py" \\
            --project-root /path/to/project \\
            --sources semantic,file
    """
    console.print(f"\n[bold]Query:[/bold] {query}\n")

    # Load configuration
    config = load_config(config_file)

    # Default project root to current directory
    if project_root is None:
        project_root = Path.cwd()

    # Parse sources
    source_list = sources.split(',') if sources else None

    # Step 1: Classify intent
    classifier = IntentClassifier()
    intent_match = classifier.classify(query)
    console.print(
        f"[bold]Intent:[/bold] {intent_match.intent.value} "
        f"(confidence: {intent_match.confidence:.2f})\n"
    )

    # Step 2: Calculate budget
    hierarchy_config = HierarchyConfig(**config.get('hierarchy', {}))
    budget_calc = BudgetCalculator(hierarchy_config)
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

    # Step 3: Collect context from sources
    console.print("[cyan]Collecting context from sources...[/cyan]")

    collector = MultiSourceCollector(
        semantic_config=config.get('sources', {}).get('semantic'),
        file_config=config.get('sources', {}).get('file'),
        cache_config=config.get('cache') if not no_cache else {'enabled': False},
    )

    candidates = collector.collect(
        query=query,
        project_root=project_root,
        current_file=current_file,
        sources=source_list,
        use_cache=not no_cache,
    )

    console.print(f"[green]✓ Collected {len(candidates)} candidates from sources[/green]\n")

    # Step 4: Initialize feedback tracker and enrich nodes
    if track_feedback:
        tracker = FeedbackTracker(**config.get('feedback', {}))
        candidates = tracker.enrich_nodes(candidates)

    # Step 5: Assign to pools
    pool_selector = PoolSelector(project_root=project_root)
    pools = pool_selector.assign_pools(candidates, query, current_file)

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

    # Step 6: Rank within pools
    console.print("[cyan]Ranking within pools...[/cyan]")

    # Embed query if any nodes need ranking
    query_embedding = None
    for pool in pools.values():
        if pool:
            # Check if any node needs embedding
            needs_embedding = any(n.embedding is None for n in pool)
            if needs_embedding:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode(query)
                # Embed nodes that don't have embeddings
                for node in pool:
                    if node.embedding is None:
                        node.embedding = model.encode(node.content)
                break

    if query_embedding is None and any(pools.values()):
        # Use a dummy model just to get query embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query)

    ranker = VectorRanker(hierarchy_config)
    if query_embedding is not None:
        ranked_pools = ranker.rank_all_pools(pools, query_embedding)
    else:
        ranked_pools = {level: [] for level in HierarchyLevel}

    # Step 7: Select within budget
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

    # Step 8: Assemble context
    assembler = ContextAssembler()
    assembled = assembler.assemble(selected, context_budget)

    console.print(f"[bold green]✓ Context optimized:[/bold green] {assembled.total_tokens} tokens")
    console.print(f"[bold]Efficiency:[/bold] {assembled.efficiency:.1%}\n")

    if verbose:
        console.print("[bold]Assembled Context Preview:[/bold]")
        preview_lines = assembled.text.split('\n')[:20]
        console.print("[dim]" + '\n'.join(preview_lines) + "\n...[/dim]\n")

        # Show feedback stats if enabled
        if track_feedback:
            console.print("[bold]Feedback Stats:[/bold]")
            stats = tracker.get_efficiency_stats(days=7)
            console.print(f"  Avg Efficiency (7d): {stats['avg_efficiency']:.1%}")
            console.print(f"  Total Queries (7d): {stats['total_queries']}")

    return assembled


@app.command()
def info():
    """Show ContextLattice configuration and status."""
    console.print("[bold]ContextLattice v0.1.0[/bold]\n")
    console.print("Query-time context optimization for agentic LLM systems\n")

    # Load config
    config = load_config()

    console.print("[bold]Hierarchy Levels:[/bold]")
    for level in HierarchyLevel:
        console.print(f"  {level.name}: {level.description}")

    console.print()

    # Show source configuration
    console.print("[bold]Configured Sources:[/bold]")
    sources_config = config.get('sources', {})
    for source_name, source_config in sources_config.items():
        enabled = source_config.get('enabled', False)
        status = "✓ Enabled" if enabled else "✗ Disabled"
        console.print(f"  {source_name}: {status}")

    console.print()

    # Show cache configuration
    cache_config = config.get('cache', {})
    cache_enabled = cache_config.get('enabled', False)
    console.print(f"[bold]Cache:[/bold] {'✓ Enabled' if cache_enabled else '✗ Disabled'}")
    if cache_enabled:
        console.print(f"  Redis: {cache_config.get('redis_url')}")
        console.print(f"  TTL: {cache_config.get('ttl')}s")

    console.print()


@app.command()
def hook(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Query to optimize (if not reading from stdin)"),
    project_root: Optional[Path] = typer.Option(None, "--project-root", "-p", help="Project root directory"),
    budget: int = typer.Option(8000, "--budget", "-b", help="Token budget for injected context"),
    sources: Optional[str] = typer.Option("semantic,file", "--sources", "-s", help="Comma-separated sources"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Config file path"),
    stdin_mode: bool = typer.Option(False, "--stdin", help="Read JSON input from stdin (Claude Code hook mode)"),
):
    """
    Run as Claude Code hook or optimize a single query.

    Hook mode (--stdin): Reads JSON from stdin, outputs context to stdout.
    Direct mode (--query): Optimizes the given query and outputs context.

    Example hook configuration in ~/.claude/settings.json:
        {
          "hooks": {
            "UserPromptSubmit": [{
              "matcher": ".*",
              "hooks": [{
                "type": "command",
                "command": "context-lattice hook --stdin",
                "timeout": 30
              }]
            }]
          }
        }

    Example direct usage:
        context-lattice hook --query "Fix the authentication bug"
    """
    import sys

    source_list = sources.split(',') if sources else ['semantic', 'file']

    pre_hook = PreQueryHook(
        config_path=config_file,
        project_root=project_root,
        budget=budget,
        sources=source_list,
    )

    if stdin_mode:
        # Claude Code hook mode - read JSON from stdin
        exit_code = pre_hook.run_from_stdin()
        raise typer.Exit(code=exit_code)

    elif query:
        # Direct mode - optimize given query
        context = pre_hook.optimize(query=query, cwd=str(project_root) if project_root else None)
        if context:
            print(context)
        raise typer.Exit(code=0)

    else:
        console.print("[red]Error: Either --query or --stdin is required[/red]")
        raise typer.Exit(code=1)


@app.command()
def install_hook(
    scope: str = typer.Option("project", "--scope", help="Install scope: 'global' (~/.claude) or 'project' (.claude)"),
    budget: int = typer.Option(8000, "--budget", "-b", help="Token budget for injected context"),
    sources: str = typer.Option("semantic,file", "--sources", "-s", help="Sources to use"),
):
    """
    Install the pre-query hook for Claude Code.

    This adds the ContextLattice hook to your Claude Code settings,
    enabling automatic context optimization for every query.

    Scopes:
        - project: Installs to .claude/settings.json (current project only)
        - global: Installs to ~/.claude/settings.json (all projects)

    Example:
        context-lattice install-hook --scope global
        context-lattice install-hook --scope project --budget 10000
    """
    import json

    if scope == "global":
        settings_path = Path.home() / ".claude" / "settings.json"
    else:
        settings_path = Path.cwd() / ".claude" / "settings.json"

    # Ensure directory exists
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing settings
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
    else:
        settings = {}

    # Add hook configuration
    if "hooks" not in settings:
        settings["hooks"] = {}

    hook_command = f"context-lattice hook --stdin --budget {budget} --sources {sources}"

    settings["hooks"]["UserPromptSubmit"] = [{
        "matcher": ".*",
        "hooks": [{
            "type": "command",
            "command": hook_command,
            "timeout": 30
        }]
    }]

    # Write updated settings
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)

    console.print(f"[green]✓ Hook installed to {settings_path}[/green]")
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Command: {hook_command}")
    console.print(f"  Budget: {budget} tokens")
    console.print(f"  Sources: {sources}")
    console.print(f"\n[dim]Restart Claude Code to activate the hook.[/dim]")


@app.command()
def uninstall_hook(
    scope: str = typer.Option("project", "--scope", help="Uninstall scope: 'global' or 'project'"),
):
    """
    Remove the pre-query hook from Claude Code settings.

    Example:
        context-lattice uninstall-hook --scope global
    """
    import json

    if scope == "global":
        settings_path = Path.home() / ".claude" / "settings.json"
    else:
        settings_path = Path.cwd() / ".claude" / "settings.json"

    if not settings_path.exists():
        console.print(f"[yellow]No settings file found at {settings_path}[/yellow]")
        return

    with open(settings_path) as f:
        settings = json.load(f)

    if "hooks" in settings and "UserPromptSubmit" in settings["hooks"]:
        del settings["hooks"]["UserPromptSubmit"]
        if not settings["hooks"]:
            del settings["hooks"]

        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)

        console.print(f"[green]✓ Hook removed from {settings_path}[/green]")
    else:
        console.print(f"[yellow]No UserPromptSubmit hook found in {settings_path}[/yellow]")


@app.command()
def test_sources(
    project_root: Optional[Path] = typer.Option(None, "--project-root", "-p", help="Project root directory"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Config file path"),
):
    """Test source connections."""
    console.print("[bold]Testing source connections...[/bold]\n")

    config = load_config(config_file)

    # Test semantic source
    console.print("[cyan]Testing Semantic Source (Qdrant)...[/cyan]")
    try:
        from ..sources import SemanticSource
        semantic_config = config.get('sources', {}).get('semantic', {})
        semantic = SemanticSource(
            qdrant_url=semantic_config.get('qdrant_url', 'http://10.32.3.27:6333'),
            collection=semantic_config.get('collection', 'caelum_knowledge'),
        )
        if semantic.test_connection():
            console.print("[green]✓ Semantic source connected[/green]\n")
        else:
            console.print("[red]✗ Semantic source connection failed[/red]\n")
    except Exception as e:
        console.print(f"[red]✗ Semantic source error: {e}[/red]\n")

    # Test file source
    console.print("[cyan]Testing File Source...[/cyan]")
    try:
        from ..sources import FileSource
        file_source = FileSource()
        project_root = project_root or Path.cwd()
        test_files = list(project_root.glob("*.py"))[:3]
        console.print(f"[green]✓ File source ready ({len(test_files)} sample files found)[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ File source error: {e}[/red]\n")

    # Test Redis cache
    console.print("[cyan]Testing Redis Cache...[/cyan]")
    try:
        cache_config = config.get('cache', {})
        if cache_config.get('enabled'):
            import redis
            redis_client = redis.from_url(cache_config.get('redis_url'), decode_responses=True)
            redis_client.ping()
            console.print("[green]✓ Redis cache connected[/green]\n")
        else:
            console.print("[yellow]○ Redis cache disabled in config[/yellow]\n")
    except Exception as e:
        console.print(f"[red]✗ Redis cache error: {e}[/red]\n")


if __name__ == "__main__":
    app()
