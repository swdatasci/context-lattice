# ContextLattice

**Query-time context optimization for agentic LLM systems**

ContextLattice solves the token budget problem for AI agents by intelligently selecting and prioritizing context based on a **semantic hierarchy** rather than just relevance scores.

## The Problem

Agentic systems (like Claude Code) need relevant context but face token limits:
- Too much context → wasted tokens, slower responses, higher costs
- Too little context → degraded quality, missing critical information
- Wrong context → poor solutions despite using tokens

## The Solution

ContextLattice implements a **hybrid hierarchy + vector approach**:

1. **Semantic Hierarchy** (4 levels) - Structural importance:
   - **STRUCTURAL**: Always included (user prefs, project context) - 15% budget
   - **DIRECT**: Query-matched (mentioned files, entities) - 45% budget
   - **IMPLIED**: Related (same module, types, tests) - 30% budget
   - **BACKGROUND**: Architectural (docs, conventions) - 10% budget

2. **Vector Ranking** within each level:
   - Embeddings for semantic similarity
   - Recency scoring (exponential decay)
   - Usage tracking (what was actually helpful)
   - User feedback boost (corrections, preferences)

3. **Cost-Aware Optimization**:
   - Free: Metadata filtering (file type, recency)
   - Cheap: Cached embeddings (Qdrant lookup)
   - Moderate: Fresh embeddings (local GPU)
   - Expensive: LLM summarization (only when needed)

## Features

- ✅ **Intent Classification**: Automatically detect task type (debugging, coding, research, etc.)
- ✅ **Budget Allocation**: Intent-specific weighting (debugging boosts DIRECT, research boosts BACKGROUND)
- ✅ **Pool Selection**: Hierarchy-based filtering before semantic search
- ✅ **Vector Ranking**: Similarity + recency + usage within pools
- ✅ **Context Assembly**: Clear, structured output with level headers
- 🚧 **Feedback Learning**: Track what context was actually used (Phase 2)
- 🚧 **Source Integration**: Semantic search, files, todos, conversations (Phase 2)
- 🚧 **Pre/Post Hooks**: Automatic context injection for Claude Code (Phase 2)

## Installation

```bash
# Using uv (recommended)
cd /home/rford/caelum/context-lattice
uv venv
source .venv/bin/activate
uv pip install -e .

# Or using pip
pip install -e .
```

## Usage

### CLI

```bash
# Optimize context for a query
context-lattice optimize --query "Fix the authentication bug in login.py"

# With custom budget
context-lattice optimize --query "..." --budget 20000

# Verbose output
context-lattice optimize --query "..." --verbose

# Show system info
context-lattice info
```

### Python API

```python
from context_lattice import (
    HierarchyConfig,
    BudgetCalculator,
    IntentClassifier,
    PoolSelector,
    VectorRanker,
    ContextAssembler,
)

# 1. Classify intent
classifier = IntentClassifier()
intent = classifier.classify("Fix the auth bug")

# 2. Calculate budget
config = HierarchyConfig()
budget_calc = BudgetCalculator(config)
budget = budget_calc.calculate(intent=intent.intent.value)

# 3. Assign to pools
pool_selector = PoolSelector()
pools = pool_selector.assign_pools(candidates, query)

# 4. Rank within pools
ranker = VectorRanker(config)
ranked = ranker.rank_all_pools(pools, query_embedding)

# 5. Select within budget
selected = {}
for level, ranked_pool in ranked.items():
    selected[level] = ranker.select_within_budget(
        ranked_pool,
        budget.per_level[level]
    )

# 6. Assemble context
assembler = ContextAssembler()
result = assembler.assemble(selected, budget)
print(result.text)
```

## Architecture

### Hierarchy Levels

```
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 0: STRUCTURAL (Always Included)                       │
│   • User preferences & corrections                          │
│   • Active project CLAUDE.md                                │
│   • Current task definition                                 │
│   └─ Budget: 15% (fixed allocation)                        │
├─────────────────────────────────────────────────────────────┤
│ LEVEL 1: DIRECT (Query-Matched)                             │
│   • Files explicitly mentioned                              │
│   • Entities referenced (functions, classes)                │
│   • Recent conversation turns                               │
│   └─ Budget: 45% (highest priority)                        │
├─────────────────────────────────────────────────────────────┤
│ LEVEL 2: IMPLIED (Semantically Related)                     │
│   • Same module/directory as direct                         │
│   • Type definitions, interfaces                            │
│   • Test files for direct code                              │
│   └─ Budget: 30% (vector ranking)                          │
├─────────────────────────────────────────────────────────────┤
│ LEVEL 3: BACKGROUND (If Space Allows)                       │
│   • Architecture docs                                       │
│   • Conventions & patterns                                  │
│   • Historical context                                      │
│   └─ Budget: 10% (summarized)                              │
└─────────────────────────────────────────────────────────────┘
```

### Query Flow

```
Query → Intent Classify → Budget Calc → Pool Selection → Vector Ranking → Assembly
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Format Code

```bash
black src/ tests/
ruff check src/ tests/
```

## Roadmap

### Phase 1: Core Engine ✅ (DONE)
- [x] Hierarchy definitions
- [x] Intent classifier
- [x] Budget calculator
- [x] Pool selector
- [x] Vector ranker
- [x] Context assembler
- [x] CLI
- [x] Unit tests

### Phase 2: Integration (Next)
- [ ] Source integrations (semantic search, files, todos)
- [ ] Feedback tracking (usage detection)
- [ ] Pre-query hook for Claude Code
- [ ] Integration tests
- [ ] Efficiency metrics

### Phase 3: Learning & Polish
- [ ] Weight updates from feedback
- [ ] User correction storage
- [ ] Summarization (Ollama)
- [ ] MCP server wrapper
- [ ] Performance benchmarks

## Configuration

Default configuration in `config/default.yaml`:

```yaml
hierarchy:
  structural_pct: 0.15
  direct_pct: 0.45
  implied_pct: 0.30
  background_pct: 0.10

  implied_threshold: 0.6
  background_threshold: 0.4

optimization:
  cache_ttl: 3600
  source_timeout: 2.0
  max_retries: 3
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Token Reduction | 30-50% |
| Context Efficiency | >60% (referenced / provided) |
| Quality Preservation | <5% degradation |
| Optimization Latency | <500ms |

## License

MIT License - see LICENSE file

## Author

Roderick Ford - [swdatasci](https://github.com/swdatasci)

## Related Projects

- [concept-graph](https://github.com/swdatasci/concept-graph) - Hierarchical concept compression
- [caelum](https://github.com/iodev/caelum) - AI infrastructure and MCP servers
- [PassiveIncomeMaximizer](https://github.com/roderickford/PassiveIncomeMaximizer) - AI trading system

---

**Status**: Phase 1 Complete (MVP)
**Version**: 0.1.0
**Created**: 2026-04-04
