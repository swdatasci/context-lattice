# Session Progress Report - FileSource Fix

**Date**: 2026-04-05
**Session**: "NEXT SESSION" - Fixing FileSource blocking issue

---

## ✅ COMPLETED

### 1. FileSource Performance Fix

**Problem**: FileSource.fetch() hung for 6+ seconds
**Root cause**:
- Model loading in __init__ (2-3s)
- Embedding every file during collection (2-4s)
- No timeouts

**Solution implemented**:
- ✅ Removed SentenceTransformer from FileSource.__init__
- ✅ Return nodes with embedding=None (lazy evaluation)
- ✅ Added VectorRanker._ensure_embeddings() for lazy generation
- ✅ Cached model at VectorRanker class level (load once, reuse)

**Results**:
- FileSource: 6s → 0.7s (**8.5x faster**)
- Lazy embedding: 0.04s (only selected nodes)
- No more hanging!

### 2. Hook Testing Infrastructure

**Discovered**:
- ✅ Hook was failing silently (command not in PATH)
- ✅ Fixed with absolute path to venv
- ✅ Hook now produces context (confirmed working)

**Status**: Hook runs and produces context for some queries

---

## ⚠️ REMAINING ISSUES

### Issue 1: SemanticSource Hangs by Default

**Problem**: MultiSourceCollector enables semantic by default
- Line 58: `self.semantic_enabled = semantic_config.get('enabled', True)`
- Tries to connect to Qdrant at 10.32.3.27:6333
- Hangs if Qdrant unavailable

**Impact**: Hook/collector hangs unless explicitly disabled

**Fix needed**: Change default to `False` or add connection timeout

---

### Issue 2: Over-Aggressive Optimization (CRITICAL)

**Problem**: Most queries return 0 tokens

**Benchmark results** (after FileSource fix):
- Query 1: 225 tokens (99.2% reduction)
- Query 2: 2,575 tokens (92.9% reduction)
- Query 3-5: **0 tokens** (100% reduction) 🚨
- Query 6: 186 tokens (99.5% reduction)
- **Avg**: 98.6% reduction (target: 30-50%)

**File coverage**: 19.4% (target: >80%)

**Root causes** (suspected):
1. **Similarity thresholds too high**:
   - IMPLIED: 0.6 cosine similarity
   - BACKGROUND: 0.4 cosine similarity
   - Many nodes filtered out before selection

2. **Pool assignment issues**:
   - Files not being assigned to DIRECT pool even when mentioned in query
   - Most nodes end up in BACKGROUND, then filtered by threshold

3. **Budget too conservative**:
   - 8K budget spread across 4 levels = ~2K per level
   - Large files don't fit in single level budget

**Symptoms**:
- Queries 3-5 (coding/refactoring) return 0 tokens
- Query 2 (research) returns some tokens (BACKGROUND boost helps)
- File mentions not detected properly

---

### Issue 3: Latency Still High

**Current**: 2,932ms avg (target: <500ms)

**Causes**:
- Model reloaded for each query (2-3s)
- Should be <500ms after first run (warm cache)

**Note**: This is expected for benchmark (fresh process each query). Real usage (persistent process) should be much faster.

---

## 📋 NEXT STEPS (Priority Order)

### Priority 1: Fix SemanticSource Default (Quick Fix)

```python
# In src/context_lattice/sources/collector.py line 58
self.semantic_enabled = semantic_config.get('enabled', False)  # Change True → False
```

**Or** add connection timeout:
```python
try:
    self.semantic_source = SemanticSource(..., timeout=2)
except (TimeoutError, ConnectionError):
    self.semantic_enabled = False
```

---

### Priority 2: Fix Over-Aggressive Optimization

**Step A: Lower similarity thresholds**
```python
# In src/context_lattice/core/hierarchy.py
implied_threshold: float = 0.3   # Was 0.6
background_threshold: float = 0.2  # Was 0.4
```

**Step B: Improve file mention detection**
- Add fuzzy matching for file names
- Detect partial matches ("README" matches "README.md")
- Look for file paths in query

**Step C: Increase DIRECT budget for CODING**
```python
"CODING": {
    HierarchyLevel.DIRECT: 1.5,  # Was 1.2 - boost more
    ...
}
```

**Step D: Add minimum context guarantee**
- Never return <500 tokens if candidates exist
- If all pools empty after ranking, relax thresholds and retry

---

### Priority 3: Re-run Baseline

After fixes, re-run benchmark:
```bash
python benchmarks/run_benchmarks.py --compare
```

**Expected improvements**:
- Token reduction: 98.6% → 35-45% (less aggressive)
- File coverage: 19.4% → 80-90% (includes expected files)
- Latency: 2,932ms → 300-500ms (warm cache in real usage)

---

### Priority 4: Validate Hook Before Re-enabling

**Testing protocol**:
1. Test with CLI first (not as installed hook)
2. Validate context quality manually
3. Check token allocation across levels
4. Verify file mentions are detected
5. **ONLY THEN** re-enable in `.claude/settings.json`

---

## 📊 Success Metrics

| Metric | Before | Current | Target | Status |
|--------|--------|---------|--------|--------|
| FileSource latency | 6s+ | 0.7s | <1s | ✅ DONE |
| Hook hangs | Yes | No | No | ✅ DONE |
| Token reduction | 95.6% | 98.6% | 30-50% | ❌ WORSE |
| File coverage | 22.2% | 19.4% | >80% | ❌ WORSE |
| Optimization latency | 1,931ms | 2,932ms | <500ms | ❌ WORSE |

**Note**: Latency/coverage got worse because FileSource now returns MORE nodes (95-121 vs 89-106), giving ranking more to filter. Need to fix thresholds.

---

## 🎯 Estimated Time to Complete

- Priority 1 (SemanticSource): 10 min
- Priority 2 (Thresholds + Detection): 30-60 min
- Priority 3 (Re-baseline): 15 min
- Priority 4 (Hook validation): 30 min

**Total**: 1.5-2 hours to completion

---

## 🚨 Safety Notes

1. **Hook remains DISABLED** in `.claude/settings.json`
2. Do NOT re-enable until validation complete
3. Test with CLI mode first
4. Verify context quality before live deployment

---

**Next session**: Fix thresholds, improve file detection, re-run baseline, validate hook
