"""
Benchmark suite for Context Lattice performance measurement.

Tracks 3 key metrics:
1. Token Reduction: How much smaller is optimized context vs raw context?
2. Context Efficiency: What % of provided context is actually used in response?
3. Optimization Latency: How long does optimization take?

Usage:
    python benchmarks/run_benchmarks.py --baseline    # Establish baseline
    python benchmarks/run_benchmarks.py --compare     # Compare to baseline
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import statistics

from context_lattice.cli.main import app
from context_lattice.sources import MultiSourceCollector
from context_lattice.retrieval import IntentClassifier, PoolSelector, VectorRanker
from context_lattice.core import HierarchyConfig, BudgetCalculator, ContextAssembler


@dataclass
class BenchmarkQuery:
    """A test query with expected characteristics."""
    query: str
    intent: str
    expected_files: List[str]
    description: str


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    query: str
    timestamp: str

    # Token metrics
    raw_context_tokens: int
    optimized_context_tokens: int
    token_reduction_pct: float

    # Efficiency metrics (requires response analysis)
    context_efficiency_pct: Optional[float]  # % of provided context referenced in response

    # Performance metrics
    optimization_latency_ms: float

    # Quality metrics
    included_expected_files: int
    total_expected_files: int
    file_coverage_pct: float


class BenchmarkRunner:
    """Runs benchmarks and tracks metrics over time."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load test queries
        self.queries = self._load_test_queries()

    def _load_test_queries(self) -> List[BenchmarkQuery]:
        """Load test queries from JSON file."""
        queries_file = Path(__file__).parent / "test_queries.json"

        if not queries_file.exists():
            # Create default test queries
            default_queries = [
                {
                    "query": "Fix the authentication bug in login.py",
                    "intent": "debugging",
                    "expected_files": ["login.py", "auth.py", "tests/test_auth.py"],
                    "description": "Specific file debugging task"
                },
                {
                    "query": "How does the semantic search work?",
                    "intent": "research",
                    "expected_files": ["semantic_source.py", "vector_ranker.py"],
                    "description": "Code understanding query"
                },
                {
                    "query": "Implement rate limiting for the API",
                    "intent": "coding",
                    "expected_files": ["api.py", "middleware.py"],
                    "description": "Feature implementation task"
                },
                {
                    "query": "Refactor the pool selector to use async",
                    "intent": "refactoring",
                    "expected_files": ["pool_selector.py", "retrieval/__init__.py"],
                    "description": "Code refactoring task"
                },
                {
                    "query": "What's the overall architecture of this project?",
                    "intent": "research",
                    "expected_files": ["README.md", "CLAUDE.md", "architecture.md"],
                    "description": "High-level architecture query"
                }
            ]

            queries_file.write_text(json.dumps(default_queries, indent=2))

        with open(queries_file) as f:
            data = json.load(f)

        return [BenchmarkQuery(**q) for q in data]

    def run_single_benchmark(self, query: BenchmarkQuery) -> BenchmarkResult:
        """Run benchmark for a single query."""
        print(f"\n{'='*60}")
        print(f"Query: {query.query}")
        print(f"Intent: {query.intent}")
        print(f"{'='*60}")

        # Step 1: Collect raw context (without optimization)
        project_root = Path.cwd()
        collector = MultiSourceCollector(
            semantic_config={'enabled': False},  # Disable semantic for baseline
            file_config={'enabled': True},
            cache_config={'enabled': False},
        )

        print("Collecting raw context...")
        raw_start = time.time()
        raw_candidates = collector.collect(
            query=query.query,
            project_root=project_root,
            sources=['file'],
            use_cache=False,
        )
        raw_context_tokens = sum(node.tokens for node in raw_candidates)
        print(f"  Raw context: {len(raw_candidates)} nodes, {raw_context_tokens} tokens")

        # Step 2: Run optimization pipeline
        print("Running optimization...")
        opt_start = time.time()

        # Classify intent
        classifier = IntentClassifier()
        intent_match = classifier.classify(query.query)

        # Calculate budget
        config = HierarchyConfig()
        budget_calc = BudgetCalculator(config)
        context_budget = budget_calc.calculate(
            conversation_tokens=0,
            tools_tokens=5000,
            intent=intent_match.intent.value,
        )

        # Assign to pools
        pool_selector = PoolSelector(project_root=project_root)
        pools = pool_selector.assign_pools(raw_candidates, query.query, None)

        # Rank and select
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query.query)

        # Ensure all nodes have embeddings
        for pool in pools.values():
            for node in pool:
                if node.embedding is None:
                    node.embedding = model.encode(node.content)

        ranker = VectorRanker(config)
        ranked_pools = ranker.rank_all_pools(pools, query_embedding)

        # Select within budget
        selected = {}
        for level in context_budget.per_level.keys():
            ranked = ranked_pools.get(level, [])
            budget_for_level = context_budget.per_level[level]
            selected[level] = ranker.select_within_budget(ranked, budget_for_level)

        # Assemble
        assembler = ContextAssembler()
        assembled = assembler.assemble(selected, context_budget)

        opt_end = time.time()
        optimization_latency_ms = (opt_end - opt_start) * 1000

        print(f"  Optimized context: {assembled.total_tokens} tokens")
        print(f"  Optimization latency: {optimization_latency_ms:.1f}ms")

        # Step 3: Calculate metrics
        token_reduction_pct = (
            (raw_context_tokens - assembled.total_tokens) / raw_context_tokens * 100
            if raw_context_tokens > 0 else 0
        )

        # Check file coverage
        included_files = set()
        for level_nodes in selected.values():
            for node in level_nodes:
                if hasattr(node, 'metadata') and 'file_path' in node.metadata:
                    included_files.add(node.metadata['file_path'])

        expected_files_found = sum(
            1 for expected in query.expected_files
            if any(expected in str(f) for f in included_files)
        )
        file_coverage_pct = (
            expected_files_found / len(query.expected_files) * 100
            if query.expected_files else 0
        )

        print(f"\n📊 Results:")
        print(f"  Token reduction: {token_reduction_pct:.1f}%")
        print(f"  File coverage: {expected_files_found}/{len(query.expected_files)} ({file_coverage_pct:.1f}%)")
        print(f"  Optimization time: {optimization_latency_ms:.1f}ms")

        return BenchmarkResult(
            query=query.query,
            timestamp=datetime.now().isoformat(),
            raw_context_tokens=raw_context_tokens,
            optimized_context_tokens=assembled.total_tokens,
            token_reduction_pct=token_reduction_pct,
            context_efficiency_pct=None,  # Requires response analysis
            optimization_latency_ms=optimization_latency_ms,
            included_expected_files=expected_files_found,
            total_expected_files=len(query.expected_files),
            file_coverage_pct=file_coverage_pct,
        )

    def run_all_benchmarks(self) -> Dict:
        """Run all benchmark queries and aggregate results."""
        results = []

        for query in self.queries:
            try:
                result = self.run_single_benchmark(query)
                results.append(asdict(result))
            except Exception as e:
                print(f"❌ Error running benchmark for '{query.query}': {e}")

        # Aggregate statistics
        if results:
            token_reductions = [r['token_reduction_pct'] for r in results]
            latencies = [r['optimization_latency_ms'] for r in results]
            coverages = [r['file_coverage_pct'] for r in results]

            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(results),
                "aggregate_metrics": {
                    "avg_token_reduction_pct": statistics.mean(token_reductions),
                    "median_token_reduction_pct": statistics.median(token_reductions),
                    "avg_optimization_latency_ms": statistics.mean(latencies),
                    "p95_optimization_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                    "avg_file_coverage_pct": statistics.mean(coverages),
                },
                "results": results
            }

            print(f"\n{'='*60}")
            print("📊 AGGREGATE RESULTS")
            print(f"{'='*60}")
            print(f"Queries tested: {summary['total_queries']}")
            print(f"Avg token reduction: {summary['aggregate_metrics']['avg_token_reduction_pct']:.1f}%")
            print(f"Avg optimization latency: {summary['aggregate_metrics']['avg_optimization_latency_ms']:.1f}ms")
            print(f"Avg file coverage: {summary['aggregate_metrics']['avg_file_coverage_pct']:.1f}%")

            return summary

        return {}

    def save_results(self, results: Dict, label: str = "baseline"):
        """Save benchmark results to file."""
        output_file = self.results_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps(results, indent=2))
        print(f"\n✅ Results saved to {output_file}")

        # Also save as "latest"
        latest_file = self.results_dir / f"{label}_latest.json"
        latest_file.write_text(json.dumps(results, indent=2))

        return output_file

    def compare_to_baseline(self, current_results: Dict):
        """Compare current results to baseline."""
        baseline_file = self.results_dir / "baseline_latest.json"

        if not baseline_file.exists():
            print("\n⚠️  No baseline found. Run with --baseline first.")
            return

        with open(baseline_file) as f:
            baseline = json.load(f)

        baseline_metrics = baseline['aggregate_metrics']
        current_metrics = current_results['aggregate_metrics']

        print(f"\n{'='*60}")
        print("📈 COMPARISON TO BASELINE")
        print(f"{'='*60}")

        # Token reduction comparison
        token_diff = current_metrics['avg_token_reduction_pct'] - baseline_metrics['avg_token_reduction_pct']
        print(f"Token reduction: {current_metrics['avg_token_reduction_pct']:.1f}% "
              f"({token_diff:+.1f}% vs baseline)")

        # Latency comparison
        latency_diff = current_metrics['avg_optimization_latency_ms'] - baseline_metrics['avg_optimization_latency_ms']
        print(f"Optimization latency: {current_metrics['avg_optimization_latency_ms']:.1f}ms "
              f"({latency_diff:+.1f}ms vs baseline)")

        # Coverage comparison
        coverage_diff = current_metrics['avg_file_coverage_pct'] - baseline_metrics['avg_file_coverage_pct']
        print(f"File coverage: {current_metrics['avg_file_coverage_pct']:.1f}% "
              f"({coverage_diff:+.1f}% vs baseline)")

        # Overall assessment
        print(f"\n{'='*60}")
        improvements = []
        regressions = []

        if token_diff > 0:
            improvements.append(f"Token reduction improved by {token_diff:.1f}%")
        elif token_diff < -5:
            regressions.append(f"Token reduction decreased by {abs(token_diff):.1f}%")

        if latency_diff < 0:
            improvements.append(f"Latency improved by {abs(latency_diff):.1f}ms")
        elif latency_diff > 100:
            regressions.append(f"Latency increased by {latency_diff:.1f}ms")

        if coverage_diff > 0:
            improvements.append(f"File coverage improved by {coverage_diff:.1f}%")
        elif coverage_diff < -10:
            regressions.append(f"File coverage decreased by {abs(coverage_diff):.1f}%")

        if improvements:
            print("✅ Improvements:")
            for imp in improvements:
                print(f"   • {imp}")

        if regressions:
            print("\n⚠️  Regressions:")
            for reg in regressions:
                print(f"   • {reg}")

        if not improvements and not regressions:
            print("➡️  No significant changes from baseline")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Context Lattice benchmarks")
    parser.add_argument("--baseline", action="store_true", help="Establish baseline metrics")
    parser.add_argument("--compare", action="store_true", help="Compare to baseline")
    parser.add_argument("--results-dir", default="benchmarks/results", help="Results directory")

    args = parser.parse_args()

    runner = BenchmarkRunner(results_dir=Path(args.results_dir))

    if args.baseline:
        print("🎯 Establishing baseline metrics...")
        results = runner.run_all_benchmarks()
        if results:
            runner.save_results(results, label="baseline")

    elif args.compare:
        print("📊 Running comparison benchmarks...")
        results = runner.run_all_benchmarks()
        if results:
            runner.save_results(results, label="comparison")
            runner.compare_to_baseline(results)

    else:
        print("❌ Error: Specify --baseline or --compare")
        parser.print_help()
