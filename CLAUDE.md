# ContextLattice - Claude Code Guide

**Query-time context optimization for agentic LLM systems**

---

## Quick Start

When starting a new session on this project:

1. **Activate environment**: `source .venv/bin/activate` (or `uv sync`)
2. **Run tests**: `pytest tests/ -v`
3. **Try CLI**: `python -m context_lattice.cli.main optimize --query "Fix the bug" --verbose`
4. **Check status**: See current phase in README.md

---

## Project Overview

**Purpose**: Solve the token budget problem for AI agents by intelligently selecting context using a semantic hierarchy + vector ranking hybrid approach.

**Key Innovation**: Not just "relevance scores" - use **structural hierarchy** to ensure critical context (user prefs, project constraints) is always included, regardless of semantic similarity.

**Current Status**: Phase 1 Complete (MVP Core Engine)

---

## Architecture

### Core Concept: Hybrid Hierarchy + Vectors

```
┌─────────────────────────────────────────────┐
│  Semantic Hierarchy (Structural Filters)    │
├─────────────────────────────────────────────┤
│  STRUCTURAL (15%) → Always include          │
│  DIRECT (45%)     → Query-matched           │
│  IMPLIED (30%)    → Same module             │
│  BACKGROUND (10%) → Architectural           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Vector Ranking (Within Pools)              │
├─────────────────────────────────────────────┤
│  Similarity × Recency × Usage × UserBoost   │
└─────────────────────────────────────────────┘
                    ↓
          Select Within Budget
```

### Module Organization

```
src/context_lattice/
├── core/               # Data structures
│   ├── hierarchy.py    # 4-level hierarchy + config
│   ├── node.py         # ContextNode with embeddings
│   ├── budget.py       # Token budget calculator
│   └── assembler.py    # Format final context
├── retrieval/          # Context selection
│   ├── intent_classifier.py  # Query intent (rule-based)
│   ├── pool_selector.py      # Assign to hierarchy pools
│   └── vector_ranker.py      # Rank within pools
└── cli/                # CLI interface
    └── main.py         # Typer-based CLI
```

---

## Key Components

### 1. HierarchyLevel (Enum)

Four levels of structural importance:
- **STRUCTURAL**: User prefs, project CLAUDE.md (always included)
- **DIRECT**: Files/entities mentioned in query
- **IMPLIED**: Same module/directory as DIRECT
- **BACKGROUND**: Docs, architecture, conventions

### 2. ContextNode (Dataclass)

```python
@dataclass
class ContextNode:
    id: str
    content: str
    tokens: int
    level: HierarchyLevel
    embedding: np.ndarray  # 384-dim
    recency_score: float   # Exponential decay
    usage_count: int       # Times referenced
    user_boost: float      # 1.5 if user-corrected
```

### 3. IntentClassifier

Rule-based pattern matching (no LLM needed):
- DEBUGGING: "fix", "bug", "error"
- CODING: "implement", "add", "create"
- RESEARCH: "how does", "where is", "what is"
- REFACTORING: "refactor", "clean up"
- PLANNING: "plan", "design", "strategy"
- DOCUMENTATION: "document", "docs"

### 4. PoolSelector

Assigns nodes to pools based on structural rules:
- STRUCTURAL: Check metadata type, file names (CLAUDE.md, user_prefs.yaml)
- DIRECT: Parse query for file mentions, entities, current file
- IMPLIED: Same directory as DIRECT nodes, related tests
- BACKGROUND: Everything else

### 5. VectorRanker

Ranks within pools using:
- `final_score = 0.7 * similarity + 0.3 * within_level_weight`
  (for IMPLIED/BACKGROUND)
- `final_score = 0.3 * similarity + 0.7 * within_level_weight`
  (for STRUCTURAL/DIRECT)

### 6. ContextAssembler

Formats final context with clear level headers:
```markdown
# STRUCTURAL CONTEXT (Always Included)
...

# DIRECT CONTEXT (Query-Matched)
...

# IMPLIED CONTEXT (Related)
...

# BACKGROUND CONTEXT (Architectural)
...
```

---

## Development Workflow

### Adding Features

1. **Add to appropriate module** (core/retrieval/optimization/feedback/sources)
2. **Write unit tests** in `tests/test_*.py`
3. **Update CLI** if user-facing
4. **Update README** with new capabilities
5. **Run full test suite**: `pytest tests/ -v`

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_hierarchy.py -v

# With coverage
pytest tests/ --cov=context_lattice --cov-report=html
```

### CLI Development

```bash
# Run CLI directly
python -m context_lattice.cli.main optimize --query "..." --verbose

# Or if installed
context-lattice optimize --query "..." --verbose
```

---

## Phases

### Phase 1: Core Engine ✅ (COMPLETE)

**Goal**: Prove the concept works

**Components**:
- [x] Hierarchy definitions (4 levels)
- [x] ContextNode dataclass
- [x] Budget calculator
- [x] Intent classifier (rule-based)
- [x] Pool selector (hierarchy filtering)
- [x] Vector ranker (cached embeddings)
- [x] Context assembler
- [x] CLI (`context-lattice optimize`)
- [x] Unit tests (hierarchy, intent)
- [x] README, CLAUDE.md

**Deliverable**: CLI that optimizes context for a query

**Test**: `context-lattice optimize --query "Fix the auth bug in login.py" --verbose`

---

### Phase 2: Integration & Feedback (NEXT)

**Goal**: Make it useful in production

**Tasks**:
- [ ] Source integrations (semantic search, files, todos, conversations)
- [ ] Feedback tracker (detect which context was referenced in response)
- [ ] Pre-query hook for Claude Code
- [ ] Cost-aware escalation (metadata → cached vectors → fresh vectors → LLM)
- [ ] Integration tests with real queries
- [ ] Efficiency metrics logging

**Deliverable**: Automatic context injection via hook

**Test**: Hook optimizes context for every Claude Code query

---

### Phase 3: Learning & Polish (FUTURE)

**Goal**: Self-improving system

**Tasks**:
- [ ] Weight updates from feedback
- [ ] User correction storage & retrieval
- [ ] Summarization for overflow (Ollama)
- [ ] MCP server wrapper
- [ ] Performance benchmarks
- [ ] Documentation polish

**Deliverable**: Production-ready MCP server

---

## Important Decisions

### Why Hybrid (Hierarchy + Vectors)?

**Option A**: Pure vector search
- Problem: Treats all context equally
- Issue: Critical context (user prefs) might get excluded if semantically dissimilar

**Option B**: Pure hierarchy
- Problem: Can't rank within levels
- Issue: All DIRECT nodes included even if irrelevant

**Option C (Chosen)**: Hybrid
- Hierarchy defines POOLS (structural importance)
- Vectors rank WITHIN pools (semantic relevance)
- Best of both: guarantees + relevance

### Why Rule-Based Intent?

Could use LLM for intent classification, but:
- Cost: Every query would need an extra API call
- Speed: Adds latency
- Accuracy: Patterns work well for code queries
- Simplicity: Easy to debug and extend

### Why Within-Level Weights?

Combining recency + usage + user boost allows:
- Recent corrections get prioritized
- Frequently referenced content boosts
- User feedback directly improves ranking
- No need for separate "correction tracking" system

---

## Configuration

Located in `src/context_lattice/core/hierarchy.py`:

```python
@dataclass
class HierarchyConfig:
    # Budget allocation
    structural_pct: float = 0.15
    direct_pct: float = 0.45
    implied_pct: float = 0.30
    background_pct: float = 0.10

    # Thresholds
    implied_threshold: float = 0.6
    background_threshold: float = 0.4

    # Intent-specific weights
    intent_weights: Dict[str, Dict[HierarchyLevel, float]]
```

**Tuning**:
- Increase `direct_pct` for debugging tasks (need buggy file)
- Increase `background_pct` for research tasks (need docs)
- Adjust thresholds to control IMPLIED/BACKGROUND inclusion

---

## Testing Strategy

### Unit Tests

Each component has isolated tests:
- `test_hierarchy.py`: Hierarchy levels, config, budget allocation
- `test_intent_classifier.py`: Intent detection, patterns, confidence
- More to come: node, pool_selector, vector_ranker, assembler

### Integration Tests (Phase 2)

End-to-end tests with real queries:
- Load sample codebase
- Run optimization pipeline
- Verify context quality and efficiency

### Performance Benchmarks (Phase 3)

Measure against targets:
- Token reduction: 30-50%
- Context efficiency: >60%
- Optimization latency: <500ms

---

## Dependencies

```toml
sentence-transformers>=2.2.0  # Embeddings (all-MiniLM-L6-v2)
qdrant-client>=1.7.0          # Vector search (Phase 2)
redis>=5.0.0                  # Caching (Phase 2)
pydantic>=2.0.0               # Data validation
typer>=0.9.0                  # CLI framework
rich>=13.0.0                  # CLI formatting
numpy>=1.24.0                 # Vector operations
```

---

## Common Tasks

### Add a New Intent Type

1. Edit `src/context_lattice/retrieval/intent_classifier.py`
2. Add to `QueryIntent` enum
3. Add patterns to `IntentClassifier.PATTERNS`
4. Add weights to `HierarchyConfig.intent_weights`
5. Add test in `tests/test_intent_classifier.py`

### Add a New Source

1. Create `src/context_lattice/sources/<source_name>.py`
2. Implement source fetching (return `List[ContextNode]`)
3. Integrate in `MultiSourceCollector` (Phase 2)
4. Add tests

### Tune Budget Allocation

1. Edit `HierarchyConfig` defaults in `hierarchy.py`
2. Or pass custom config to `BudgetCalculator`
3. Run tests to verify allocations sum to 1.0

---

## Integration with Caelum

ContextLattice will integrate with existing Caelum infrastructure:

**Data Sources**:
- Semantic search: `search_caelum_knowledge` MCP tool
- Files: Direct file reads via `Read` tool
- Todos: TaskList tool
- Conversations: History from `.claude/history.jsonl`

**Storage**:
- Embeddings: Qdrant (10.32.3.27:6333)
- Feedback: Redis (10.32.3.27:6379)
- Corrections: MongoDB (caelum-unified)

**Hooks**:
- Pre-query: `~/.claude/hooks/pre-prompt` → optimize context
- Post-query: `~/.claude/hooks/post-response` → track usage

---

## Troubleshooting

### ImportError: No module named 'context_lattice'

```bash
# Install in editable mode
cd /home/rford/caelum/context-lattice
uv pip install -e .
```

### sentence-transformers model download slow

First run downloads ~400MB model. Subsequent runs use cache.

### Tests fail with embedding errors

Make sure sentence-transformers is installed:
```bash
uv pip install sentence-transformers
```

---

## Next Session Checklist

When resuming work:

1. **Check current phase**: See README.md status
2. **Review task list**: `TaskList` to see pending work
3. **Run tests**: `pytest tests/ -v` to verify nothing broken
4. **Check git status**: `git status` for uncommitted changes
5. **Read NEXT_STEPS.md** (will be created at end of each phase)

---

**Last Updated**: 2026-04-04
**Current Phase**: Phase 1 Complete
**Next Phase**: Phase 2 (Integration & Feedback)
