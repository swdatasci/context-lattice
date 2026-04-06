# Context Lattice Benchmarks

## Overview

This benchmark suite tracks 3 key performance metrics:

| Metric | Target | Description |
|--------|--------|-------------|
| **Token Reduction** | 30-50% | How much smaller is optimized context vs raw context? |
| **Context Efficiency** | >60% | What % of provided context is referenced in response? |
| **Optimization Latency** | <500ms | How long does optimization take? |

## Usage

### Establish Baseline

Run this ONCE at the start to establish baseline metrics:

```bash
cd /home/rford/caelum/context-lattice
source .venv/bin/activate
python benchmarks/run_benchmarks.py --baseline
```

This creates:
- `benchmarks/results/baseline_YYYYMMDD_HHMMSS.json` - Timestamped baseline
- `benchmarks/results/baseline_latest.json` - Latest baseline (for comparison)

### Run Comparison

After making changes, run comparison to see improvements:

```bash
python benchmarks/run_benchmarks.py --compare
```

This:
- Runs the same test queries
- Compares to baseline
- Shows improvements/regressions
- Saves as `comparison_YYYYMMDD_HHMMSS.json`

## Test Queries

Located in `test_queries.json`, includes representative queries for:

- **Debugging**: "Fix the authentication bug in login.py"
- **Research**: "How does the semantic search work?"
- **Coding**: "Implement rate limiting for the API"
- **Refactoring**: "Refactor the pool selector to use async"
- **Architecture**: "What's the overall architecture of this project?"
- **Testing**: "Add unit tests for the VectorRanker class"

## Metrics Explained

### Token Reduction

```
Token Reduction % = (Raw Tokens - Optimized Tokens) / Raw Tokens × 100
```

**Raw Context**: All files collected without optimization
**Optimized Context**: Files selected by Context Lattice hierarchy + vectors

**Target**: 30-50% reduction
- Too high (>80%): Might be excluding important context
- Too low (<20%): Not optimizing enough

### Context Efficiency

```
Context Efficiency % = Chunks Referenced in Response / Total Chunks Provided × 100
```

**Requires**: Response analysis (Phase 3 feature)
**Measured by**: Tracking which context chunks appear in Claude's response

**Target**: >60% efficiency
- High efficiency = Most provided context was useful
- Low efficiency = Wasting tokens on unused context

### Optimization Latency

```
Optimization Latency = Time to run full pipeline (classify + pool + rank + select)
```

**Target**: <500ms
- <200ms: Excellent (barely noticeable)
- 200-500ms: Good (acceptable delay)
- >500ms: Needs optimization

**Note**: First run is slower due to model loading (~2-3s). Subsequent runs should hit target.

## Benchmark Results

### Baseline (2026-04-05)

Initial baseline established with Phase 3 infrastructure:

| Metric | Value | Status |
|--------|-------|--------|
| Avg Token Reduction | TBD | 🎯 Targeting 30-50% |
| Avg Optimization Latency | TBD | 🎯 Targeting <500ms |
| Avg File Coverage | TBD | 🎯 Targeting >80% |

**Note**: Baseline will be established in this session.

### Future Comparisons

Track improvements over time:

```
📈 COMPARISON TO BASELINE
============================================================
Token reduction: 45.2% (+5.3% vs baseline) ✅
Optimization latency: 320ms (-180ms vs baseline) ✅
File coverage: 85.0% (+5.0% vs baseline) ✅
```

## Adding New Test Queries

Edit `test_queries.json`:

```json
{
  "query": "Your test query here",
  "intent": "debugging|coding|research|refactoring|planning|documentation",
  "expected_files": ["file1.py", "file2.py"],
  "description": "Brief description of what this tests"
}
```

**Tips**:
- Cover different intent types
- Mix specific files with general queries
- Include edge cases (very long queries, ambiguous queries)
- Test with real queries from actual usage

## Interpreting Results

### Good Results ✅

```
Token reduction: 40-50%  (sweet spot)
Optimization latency: 200-400ms (fast enough)
File coverage: 80-100% (got what we needed)
```

### Warning Signs ⚠️

```
Token reduction: >80% (might be over-aggressive)
Token reduction: <20% (not optimizing enough)
Latency: >1000ms (too slow for interactive use)
File coverage: <50% (missing critical files)
```

### Regressions 🚨

If comparison shows:
- Token reduction decreased significantly
- Latency increased by >200ms
- File coverage dropped >20%

**Action**: Investigate recent changes, revert if necessary.

## Integration with Claude Code Hook

The hook is configured to run automatically in `.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "matcher": ".*",
      "hooks": [{
        "type": "command",
        "command": "context-lattice hook --stdin --budget 8000 --sources semantic,file",
        "timeout": 30
      }]
    }]
  }
}
```

**To verify hook is active**:
1. Check settings: `cat .claude/settings.json`
2. Restart Claude Code session
3. Hook runs automatically on every query

**To see hook impact**:
- Compare metrics with hook enabled vs disabled
- Check if token reduction matches benchmark results
- Monitor optimization latency in practice

## Cost Impact

**Goal**: Optimize without adding cost

| Operation | Cost | Time |
|-----------|------|------|
| Optimization pipeline | $0 (local) | <500ms |
| Embedding generation | $0 (local Ollama) | <200ms |
| Vector search | $0 (local Qdrant) | <50ms |
| Claude API call | ~$0.003/1K tokens | Depends on model |

**Savings calculation**:

```
Tokens saved per query = Raw Tokens × Reduction %
Cost saved per query = Tokens saved × $0.003 / 1000

Example (40% reduction on 10K raw context):
Saved = 10,000 × 0.40 = 4,000 tokens
Cost savings = 4,000 × $0.003 / 1000 = $0.012 per query
```

At 100 queries/day: **$1.20/day saved** (~$438/year)

## Next Steps

1. ✅ Establish baseline (this session)
2. ⏳ Implement cost-aware escalation
3. ⏳ Add context efficiency tracking (requires response analysis)
4. ⏳ Run comparisons after each optimization
5. ⏳ Track metrics over time in InfluxDB

---

**Last Updated**: 2026-04-05
**Baseline Status**: In progress
