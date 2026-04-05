# Phase 2: Integration & Feedback

**Status**: In Progress
**Started**: 2026-04-04
**Goal**: Make ContextLattice useful in production with real data sources

---

## Objectives

1. **Source Integrations** - Fetch real context from multiple sources
2. **Feedback Tracking** - Learn from what context is actually used
3. **Resilience** - Graceful degradation, caching, error handling
4. **Production Ready** - Integration tests, metrics, documentation

---

## Tasks

### 1. Source Integrations

#### Semantic Search Source
- **File**: `src/context_lattice/sources/semantic_source.py`
- **Integration**: Caelum's `search_caelum_knowledge` MCP tool + direct Qdrant
- **Returns**: `List[ContextNode]` with pre-computed embeddings from vector DB
- **Features**:
  - Query Qdrant at 10.32.3.27:6333
  - Filter by project, recency, type
  - Use existing triple-model ensemble embeddings
  - Map results to ContextNode with metadata

#### File Source
- **File**: `src/context_lattice/sources/file_source.py`
- **Integration**: Direct filesystem reads
- **Returns**: `List[ContextNode]` from files in current project
- **Features**:
  - Extract functions/classes (AST parsing for Python/TypeScript)
  - Assign hierarchy levels based on file path patterns
  - Track file modification times for recency
  - Handle multiple file types (.py, .ts, .md, .yaml)

#### Todo Source
- **File**: `src/context_lattice/sources/todo_source.py`
- **Integration**: TaskList tool (if available)
- **Returns**: `List[ContextNode]` from active tasks
- **Features**:
  - Fetch pending/in-progress tasks
  - Assign to STRUCTURAL or DIRECT based on relevance
  - Include task descriptions and context

#### Conversation Source
- **File**: `src/context_lattice/sources/conversation_source.py`
- **Integration**: `.claude/history.jsonl` or conversation API
- **Returns**: `List[ContextNode]` from recent conversation
- **Features**:
  - Parse recent conversation turns (last 3-5)
  - Assign to DIRECT level (always relevant to current query)
  - Extract code blocks, file mentions

### 2. Multi-Source Collector

- **File**: `src/context_lattice/sources/collector.py`
- **Orchestrates**: All source integrations in parallel
- **Features**:
  - Async/parallel fetching from all sources
  - Graceful degradation (continue if source fails)
  - Redis caching (cache key: hash(query + sources))
  - Circuit breakers per source
  - Deduplication (same content from multiple sources)
  - Merge metadata from different sources

### 3. Feedback Tracking

- **File**: `src/context_lattice/feedback/tracker.py`
- **Purpose**: Learn from what context was actually useful
- **Features**:
  - **Reference Detection**: Parse response to find which context was used
    - File path mentions
    - Function/class name references
    - Code snippet overlap
  - **Usage Increment**: Update `ContextNode.usage_count` in Redis
  - **User Corrections**: Store corrections with high boost
  - **Efficiency Metrics**: Track referenced/provided ratio
  - **Storage**: Redis with TTL (e.g., 30 days)

### 4. Cost-Aware Escalation

- **File**: `src/context_lattice/optimization/cost_levels.py`
- **Purpose**: Only use expensive operations when needed
- **Levels**:
  - **Level 0 (Free)**: Metadata filtering, file type checks, recency
  - **Level 1 (Cheap)**: Cached embeddings from Qdrant
  - **Level 2 (Moderate)**: Fresh embeddings (local GPU)
  - **Level 3 (Expensive)**: LLM summarization (Ollama local, Claude API last resort)
- **Strategy**: Start cheap, escalate only if budget not met

### 5. CLI Integration

- **Update**: `src/context_lattice/cli/main.py`
- **Changes**:
  - Replace demo candidates with real `MultiSourceCollector`
  - Add Redis connection for caching and feedback
  - Add `--sources` flag to select sources (semantic, files, todos, conversations)
  - Add `--no-cache` flag to bypass cache
  - Add feedback tracking option
  - Show efficiency metrics

### 6. Integration Tests

- **File**: `tests/test_integration.py`
- **Tests**:
  - Real query on actual codebase
  - Multi-source collection
  - Feedback tracking
  - Cache hit/miss
  - Graceful degradation (source failures)
  - Efficiency metrics validation

### 7. Configuration

- **File**: `config/default.yaml`
- **Settings**:
  ```yaml
  sources:
    semantic:
      enabled: true
      qdrant_url: "http://10.32.3.27:6333"
      collection: "caelum_knowledge"
      limit: 10
    files:
      enabled: true
      max_files: 20
      file_types: [".py", ".ts", ".tsx", ".js", ".md", ".yaml"]
    todos:
      enabled: false  # Optional
    conversations:
      enabled: true
      max_turns: 5

  cache:
    enabled: true
    redis_url: "redis://10.32.3.27:6379"
    ttl: 3600  # 1 hour

  feedback:
    enabled: true
    redis_url: "redis://10.32.3.27:6379"
    usage_ttl: 2592000  # 30 days
  ```

---

## Implementation Order

### Session 1 (Current): Sources
1. ✅ Create directory structure
2. ✅ Implement semantic search source
3. ✅ Implement file source
4. ✅ Implement multi-source collector
5. ✅ Add configuration loading

### Session 2: Feedback & Testing
1. Implement feedback tracker
2. Update CLI for real sources
3. Add integration tests
4. Test on real codebase (context-lattice itself)

### Session 3: Polish & Documentation
1. Add cost-aware escalation
2. Add pre-query hook support
3. Update documentation
4. Performance benchmarks

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Source Integration | 4/4 working | semantic, files, todos, conversations |
| Feedback Accuracy | >80% | Referenced nodes / total provided |
| Cache Hit Rate | >50% | Cache hits / total queries |
| Source Availability | >95% | Successful fetches / attempts |
| Integration Tests | 100% passing | All test cases green |

---

## Dependencies

```toml
# Additional for Phase 2
dependencies = [
    # ... existing ...
    "redis>=5.0.0",           # Caching and feedback storage
    "qdrant-client>=1.7.0",   # Direct Qdrant access
    "aiofiles>=23.0.0",       # Async file operations
    "watchdog>=3.0.0",        # File watching (optional)
]
```

---

## Risks & Mitigations

**Risk**: Qdrant connection fails
- **Mitigation**: Graceful degradation, fall back to file source only

**Risk**: Redis unavailable
- **Mitigation**: Disable caching/feedback, continue with fresh fetches

**Risk**: File parsing errors
- **Mitigation**: Try-catch per file, continue with others

**Risk**: Slow multi-source collection
- **Mitigation**: Parallel fetching, timeouts, cache aggressively

---

**Status**: Starting Session 1
**Next**: Implement semantic search source
