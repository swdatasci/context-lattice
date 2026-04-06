# Manual Quality Review Guide

## Goal

Validate that the hook provides **useful, relevant context** for different query types.

---

## Test Queries

Here are 5 real-world queries to test:

### Query 1: Code Understanding
```
How does the VectorRanker lazy embedding work?
```
**Expected**: Should include `vector_ranker.py`, maybe `file_source.py`

### Query 2: Architecture
```
What's the overall project structure?
```
**Expected**: Should include `README.md`, `CLAUDE.md`, maybe architecture files

### Query 3: Debugging
```
Why did FileSource hang before the fix?
```
**Expected**: Should include `file_source.py`, maybe `CRITICAL_BUG_REPORT.md`

### Query 4: Implementation
```
How do I run the benchmarks?
```
**Expected**: Should include `benchmarks/README.md`, `run_benchmarks.py`

### Query 5: Specific File
```
What does pool_selector.py do?
```
**Expected**: Should include `pool_selector.py`, related files in `retrieval/`

---

## Testing Procedure

### Step 1: Test WITHOUT Hook (Baseline)

Just ask Claude Code the query normally. Note what information you need to provide or what Claude asks for.

**Example**:
```
You: "How does the VectorRanker lazy embedding work?"
Claude: "Let me read the VectorRanker file..."
[Claude uses Read tool, explores on its own]
```

### Step 2: Test WITH Hook (Simulated)

Use the CLI to see what context the hook would inject:

```bash
cd /home/rford/caelum/context-lattice
source .venv/bin/activate

# Test the query
echo '{"query": "How does the VectorRanker lazy embedding work?", "cwd": "'$(pwd)'", "user_prompt": "How does the VectorRanker lazy embedding work?"}' | \
  /home/rford/caelum/context-lattice/.venv/bin/context-lattice hook --stdin --budget 8000 --sources file 2>/dev/null | head -100
```

**What to look for**:
- ✅ Includes relevant files
- ✅ Context is readable and organized
- ✅ Enough detail to answer the query
- ❌ Too much irrelevant content
- ❌ Missing critical files

### Step 3: Quality Assessment

For each query, rate the injected context:

| Rating | Criteria |
|--------|----------|
| **Excellent** | Includes all needed files, well-organized, could answer query with this alone |
| **Good** | Includes most needed files, minor gaps, mostly useful |
| **OK** | Includes some relevant files, but missing key info |
| **Poor** | Mostly irrelevant or missing critical files |

---

## Quick Test Script

I've created a test script for you:

```bash
cd /home/rford/caelum/context-lattice
source .venv/bin/activate

# Test all 5 queries
for query in \
  "How does the VectorRanker lazy embedding work?" \
  "What is the overall project structure?" \
  "Why did FileSource hang before the fix?" \
  "How do I run the benchmarks?" \
  "What does pool_selector.py do?"
do
  echo ""
  echo "========================================"
  echo "QUERY: $query"
  echo "========================================"
  echo ""

  # Show context that would be injected
  echo '{"query": "'"$query"'", "cwd": "'$(pwd)'", "user_prompt": "'"$query"'"}' | \
    /home/rford/caelum/context-lattice/.venv/bin/context-lattice hook --stdin --budget 8000 --sources file 2>/dev/null | head -80

  echo ""
  echo "----------------------------------------"
  echo "ASSESSMENT: [ ] Excellent [ ] Good [ ] OK [ ] Poor"
  echo "NOTES:"
  echo ""
  read -p "Press Enter for next query..."
done
```

---

## What Good Context Looks Like

### Example: "How does VectorRanker lazy embedding work?"

**Good context should include**:
```
# STRUCTURAL CONTEXT (Always Included)
[Project CLAUDE.md - project guidelines]

# DIRECT CONTEXT (Query-Matched)
## /path/to/vector_ranker.py
class VectorRanker:
    """
    Rank context nodes within hierarchy pools using vectors.

    Lazy embedding: Only generates embeddings for nodes that need ranking.
    Model is cached at class level and reused across instances.
    """

    _model_cache = None  # Class-level cache

    def _ensure_embeddings(self, nodes):
        """Ensure all nodes have embeddings (lazy generation)."""
        nodes_to_embed = [node for node in nodes if node.embedding is None]
        if not nodes_to_embed:
            return
        ...

# IMPLIED CONTEXT (Related)
## /path/to/file_source.py
[Shows that FileSource returns nodes with embedding=None]
...

# BACKGROUND CONTEXT (Architectural)
[Maybe hierarchy.py showing the overall structure]
```

**Bad context would be**:
- Just the README (too high-level)
- Unrelated files (semantic_source.py when asking about VectorRanker)
- Too little (only class definition, no implementation)

---

## Comparison Template

For each query, document:

```
QUERY: [The question]

WITHOUT HOOK:
- What tools did Claude need to use? (Read, Glob, Grep, etc.)
- How many tool calls?
- Did it find the right files?

WITH HOOK (CLI output):
- Files included: [list]
- Tokens: [number]
- Quality rating: [Excellent/Good/OK/Poor]
- Issues: [any problems]

VERDICT:
[ ] Hook helps (provides useful context upfront)
[ ] Hook neutral (doesn't hurt but doesn't help much)
[ ] Hook hurts (injects irrelevant/confusing context)
```

---

## Success Criteria

Hook is ready for deployment if:

✅ **4/5 queries rated "Good" or better**
✅ **0/5 queries rated "Poor"**
✅ **Subjective feel**: "This context would help me answer the query"

---

## After Testing

Document results in a new file `QUALITY_REVIEW_RESULTS.md`:

```markdown
# Quality Review Results

**Date**: 2026-04-05
**Reviewer**: [Your name]

## Summary

- Excellent: X/5
- Good: X/5
- OK: X/5
- Poor: X/5

**Overall verdict**: [Ready / Needs work / Not ready]

## Per-Query Results

[Your detailed notes]

## Recommendation

[Deploy / Fix issues first / Needs redesign]
```

---

## Quick One-Line Test

If you just want to see context for a single query quickly:

```bash
source .venv/bin/activate && echo '{"query": "YOUR QUERY HERE", "cwd": "'$(pwd)'", "user_prompt": "YOUR QUERY HERE"}' | context-lattice hook --stdin --budget 8000 --sources file 2>/dev/null
```

Example:
```bash
source .venv/bin/activate && echo '{"query": "How does VectorRanker work?", "cwd": "'$(pwd)'", "user_prompt": "How does VectorRanker work?"}' | context-lattice hook --stdin --budget 8000 --sources file 2>/dev/null
```
