# CRITICAL BUG: FileSource Hangs Indefinitely

**Date**: 2026-04-05
**Severity**: 🚨 BLOCKING - Makes hooks unusable
**Status**: IDENTIFIED - Needs fix

---

## Summary

`FileSource.fetch()` hangs indefinitely during execution, causing:
- Hook timeouts (30s configured, source never returns)
- Empty context injection
- Baseline test failures (0 tokens returned)
- **Hook appears to work but does nothing**

## Reproduction

```bash
cd /home/rford/caelum/context-lattice
source .venv/bin/activate

timeout 10 python -c "
from context_lattice.sources import FileSource
from pathlib import Path

fs = FileSource()
nodes = fs.fetch(
    query='README',
    project_root=Path('.'),
    max_files=5
)
print(f'Found {len(nodes)} nodes')
"
```

**Result**: Times out after 10s, never returns

## Root Cause (Suspected)

Looking at `/home/rford/caelum/context-lattice/src/context_lattice/sources/file_source.py`:

### Issue 1: Model Loading in __init__

```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2", ...):
    self.model = SentenceTransformer(model_name)  # BLOCKING
```

**Problem**: Loads 103-weight model synchronously on every instantiation.

**Impact**: 2-3 second delay EACH TIME FileSource is created.

### Issue 2: Embedding Every File

```python
def _process_file(self, file_path, query):
    # ... read file ...
    embedding = self.model.encode(content)  # BLOCKING for each file
```

**Problem**: Embeds every file's content during collection.

**Impact**: 100-200ms per file × 20 files = 2-4 seconds

### Issue 3: No Timeout on File I/O

```python
def _find_relevant_files(self, query, project_root, current_file, max_files):
    for file_path in all_files:
        # No timeout on reads
        ...
```

**Problem**: If a file is very large or slow to read, hangs indefinitely.

### Issue 4: Compound Effect

Total latency: Model loading (2s) + File reads (1s) + Embeddings (3s) = **6+ seconds**

With 30s hook timeout, sometimes completes, sometimes times out.

## Why Hook Appears to Work

1. Hook configured in `.claude/settings.json` with 30s timeout
2. FileSource hangs for 30+ seconds
3. Claude Code times out the hook
4. Hook returns exit code 1 (error)
5. **Claude Code ignores error and proceeds without injected context**
6. User sees queries work normally (because no context injected)

**False positive**: "Hook works because queries succeed" ← Actually hook is failing silently!

## Baseline Test Results Explained

From `BASELINE_REPORT.md`:

| Query | Optimized Tokens | Issue |
|-------|-----------------|-------|
| Query 1 | 225 tokens | FileSource timed out, minimal context |
| Query 2 | 4,304 tokens | Partial success |
| Query 3 | **0 tokens** | Complete timeout, no context |

**Avg latency**: 1,931ms (should be <500ms)

## Impact Assessment

### Current State
- ❌ Hook does not inject context (fails silently)
- ❌ Baseline metrics unreliable (timeouts, not optimization issues)
- ❌ Cannot measure true token reduction
- ❌ Cannot validate Phase 3 goals

### User Impact
- ✅ **No negative impact on queries** (hook failing = same as disabled)
- ⚠️  **False sense that "everything works"**
- ❌ **No benefit from optimization** (not actually running)

## Fix Strategy

### Priority 1: Make FileSource Non-Blocking (URGENT)

**Option A: Remove embeddings from FileSource** (RECOMMENDED)
```python
def _process_file(self, file_path, query):
    # Return nodes WITHOUT embeddings
    # Let VectorRanker embed only selected nodes
    return ContextNode(..., embedding=None)
```

**Benefits**:
- Instant file collection
- Lazy embedding (only when needed for ranking)
- Matches baseline test approach (disabled semantic)

**Option B: Cache embeddings in Qdrant**
- First run: Generate + cache
- Subsequent: Retrieve from cache
- Fallback: Return without embedding

**Option C: Use async/multiprocessing**
- Parallel file reads
- Parallel embedding generation
- Requires async rewrite

### Priority 2: Add Model Caching

```python
class FileSource:
    _model_cache = {}  # Class-level cache

    @classmethod
    def _get_model(cls, model_name):
        if model_name not in cls._model_cache:
            cls._model_cache[model_name] = SentenceTransformer(model_name)
        return cls._model_cache[model_name]

    def __init__(self, model_name="all-MiniLM-L6-v2", ...):
        self.model = self._get_model(model_name)  # Reuse cached model
```

### Priority 3: Add Timeouts

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def fetch(self, query, project_root, max_files=20):
    try:
        with timeout(5):  # 5s max for file collection
            nodes = self._collect_files(...)
            return nodes
    except TimeoutError:
        logger.warning("FileSource timed out, returning partial results")
        return partial_results
```

## Recommended Fix (Quick Win)

**Step 1**: Remove embedding from FileSource (Option A)
- File: `src/context_lattice/sources/file_source.py`
- Change: `return ContextNode(..., embedding=None)`
- Impact: FileSource becomes instant (<100ms)

**Step 2**: Move embedding to VectorRanker
- Only embed nodes that need ranking
- Skip STRUCTURAL nodes (no ranking needed)
- Lazy + selective = much faster

**Step 3**: Cache model at module level
- Load model once per process
- Reuse across FileSource instances

**Expected improvement**:
- FileSource: 6s → 100ms (60x faster)
- Hook latency: 2s → 300ms (meets <500ms target)
- Hook success rate: ~50% → 95%

## Testing After Fix

```bash
# Test 1: FileSource speed
time python -c "
from context_lattice.sources import FileSource
nodes = FileSource().fetch(query='test', project_root=Path('.'))
print(f'{len(nodes)} nodes')
"
# Expected: <500ms, returns nodes

# Test 2: Hook end-to-end
echo '{"query": "test", "cwd": "."}' | \
  time /path/to/context-lattice hook --stdin
# Expected: <1s, outputs context

# Test 3: Baseline re-run
python benchmarks/run_benchmarks.py --compare
# Expected: All queries complete, latency <500ms avg
```

## Verification Checklist

- [ ] FileSource completes in <500ms
- [ ] Hook injects context (verify with test query)
- [ ] Baseline tests complete without timeouts
- [ ] Token reduction metrics stabilize
- [ ] File coverage improves (nodes actually returned)

## Next Steps

1. **Implement Option A** (remove embeddings from FileSource)
2. **Add model caching** (class-level cache)
3. **Re-run baseline** with fixed FileSource
4. **Update hook config** once verified working
5. **Document** new baseline metrics

---

**Critical**: This bug makes ALL Phase 3 metrics unreliable. Must fix before continuing performance optimization.
