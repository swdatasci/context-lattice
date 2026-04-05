"""
Pre-Query Hook for Claude Code.

This hook intercepts user prompts and injects optimized context.

Usage:
    As Claude Code hook (reads JSON from stdin, outputs context to stdout):
        echo '{"user_prompt": "Fix the bug"}' | python -m context_lattice.hooks.pre_query

    From CLI:
        context-lattice hook --query "Fix the bug"

The hook outputs optimized context to stdout which Claude Code injects
as a system reminder before processing the user's query.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..core import HierarchyConfig, BudgetCalculator, ContextAssembler, HierarchyLevel
from ..retrieval import IntentClassifier, PoolSelector, VectorRanker
from ..sources import MultiSourceCollector

# Configure logging to stderr (not stdout - stdout is for context output)
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


@dataclass
class HookInput:
    """Parsed input from Claude Code hook event."""
    session_id: str
    cwd: str
    user_prompt: str
    hook_event_name: str
    raw_data: Dict[str, Any]

    @classmethod
    def from_stdin(cls) -> 'HookInput':
        """Parse hook input from stdin JSON."""
        try:
            raw = sys.stdin.read()
            if not raw.strip():
                raise ValueError("Empty stdin")

            data = json.loads(raw)

            # UserPromptSubmit event structure
            return cls(
                session_id=data.get('session_id', ''),
                cwd=data.get('cwd', str(Path.cwd())),
                user_prompt=data.get('user_prompt', data.get('prompt', '')),
                hook_event_name=data.get('hook_event_name', 'UserPromptSubmit'),
                raw_data=data,
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from stdin: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse hook input: {e}")
            raise


class PreQueryHook:
    """
    Pre-query hook that optimizes context before Claude processes the query.

    When invoked by Claude Code's UserPromptSubmit hook:
    1. Receives query in JSON format on stdin
    2. Runs ContextLattice optimization
    3. Outputs optimized context to stdout
    4. Claude Code injects this as a system reminder

    This provides automatic, zero-effort context optimization for every query.
    """

    # Class-level model cache (loaded once, reused across instances)
    _embedding_model = None

    def __init__(
        self,
        config_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
        budget: int = 8000,  # Conservative default for hook injection
        sources: Optional[list] = None,
    ):
        self.config_path = config_path
        self.project_root = project_root
        self.budget = budget
        self.sources = sources or ['semantic', 'file']
        self.config = self._load_config()

    @classmethod
    def _get_embedding_model(cls):
        """Get cached embedding model (lazy initialization)."""
        if cls._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        return cls._embedding_model

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        import yaml

        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)

        # Default config location
        default_path = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
        if default_path.exists():
            with open(default_path) as f:
                return yaml.safe_load(f)

        return {}

    def optimize(self, query: str, cwd: Optional[str] = None) -> str:
        """
        Optimize context for the given query.

        Args:
            query: User's query text
            cwd: Current working directory (for file source)

        Returns:
            Optimized context string ready for injection
        """
        project_root = Path(cwd) if cwd else self.project_root or Path.cwd()

        # Quick intent classification (no API call, rule-based)
        classifier = IntentClassifier()
        intent_match = classifier.classify(query)

        # Calculate budget allocation
        # To get our desired budget, we calculate "fake" conversation tokens
        # Formula: available = MAX_TOKENS - conversation_tokens - RESERVED_FOR_RESPONSE
        # So: conversation_tokens = MAX_TOKENS - desired_budget - RESERVED_FOR_RESPONSE
        from ..core.budget import MAX_TOKENS, RESERVED_FOR_RESPONSE
        fake_conversation_tokens = MAX_TOKENS - self.budget - RESERVED_FOR_RESPONSE

        hierarchy_config = HierarchyConfig(**self.config.get('hierarchy', {}))
        budget_calc = BudgetCalculator(hierarchy_config)
        context_budget = budget_calc.calculate(
            conversation_tokens=fake_conversation_tokens,
            intent=intent_match.intent.value,
        )

        # Collect context from sources
        collector = MultiSourceCollector(
            semantic_config=self.config.get('sources', {}).get('semantic'),
            file_config=self.config.get('sources', {}).get('file'),
            cache_config=self.config.get('cache', {'enabled': False}),  # Disable cache for speed
        )

        try:
            candidates = collector.collect(
                query=query,
                project_root=project_root,
                sources=self.sources,
                use_cache=False,  # Fresh results for hooks
            )
        except Exception as e:
            logger.warning(f"Source collection failed: {e}")
            candidates = []

        if not candidates:
            # No context to inject
            return ""

        # Assign to hierarchy pools
        pool_selector = PoolSelector(project_root=project_root)
        pools = pool_selector.assign_pools(candidates, query)

        # Embed query for ranking (using cached model)
        query_embedding = None
        model = self._get_embedding_model()
        if model is not None:
            try:
                query_embedding = model.encode(query)

                # Embed candidates that need it
                for pool in pools.values():
                    for node in pool:
                        if node.embedding is None:
                            node.embedding = model.encode(node.content)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")

        # Rank within pools
        ranker = VectorRanker(hierarchy_config)
        if query_embedding is not None:
            ranked_pools = ranker.rank_all_pools(pools, query_embedding)
        else:
            ranked_pools = pools

        # Select within budget
        selected = {}
        for level in HierarchyLevel:
            ranked = ranked_pools.get(level, [])
            budget_for_level = context_budget.per_level[level]
            selected[level] = ranker.select_within_budget(ranked, budget_for_level)

        # Assemble final context
        assembler = ContextAssembler()
        assembled = assembler.assemble(selected, context_budget)

        return assembled.text

    def run_from_stdin(self) -> int:
        """
        Main entry point for Claude Code hook.

        Reads JSON from stdin, optimizes context, outputs to stdout.
        Returns exit code (0 = success, 2 = block, other = error).
        """
        try:
            hook_input = HookInput.from_stdin()

            if not hook_input.user_prompt:
                # No prompt to optimize
                return 0

            # Optimize context
            context = self.optimize(
                query=hook_input.user_prompt,
                cwd=hook_input.cwd,
            )

            if context:
                # Output context to stdout - Claude Code injects this
                print(context)

            return 0

        except json.JSONDecodeError:
            # Invalid JSON input - don't block, just log
            logger.error("Invalid JSON input")
            return 1

        except Exception as e:
            # Log error but don't block user's query
            logger.error(f"Hook error: {e}")
            return 1


def main():
    """CLI entry point for hook."""
    hook = PreQueryHook()
    exit_code = hook.run_from_stdin()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
