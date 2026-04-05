"""
Query intent classification for context weighting.

Uses rule-based pattern matching to classify queries into task types:
- CODING: Writing new code
- DEBUGGING: Fixing bugs
- REFACTORING: Restructuring code
- RESEARCH: Understanding codebase
- PLANNING: Architecture, design
- DOCUMENTATION: Writing docs
- UNKNOWN: Fallback

Intent determines hierarchy-level budget weighting (e.g., debugging boosts
DIRECT level, research boosts BACKGROUND level).
"""

import re
from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass


class QueryIntent(Enum):
    """
    Query intent categories for context optimization.

    Each intent has different context priorities:
    - DEBUGGING → prioritize buggy files, recent changes
    - RESEARCH → prioritize docs, architecture
    - CODING → balanced approach
    - etc.
    """

    CODING = "CODING"
    DEBUGGING = "DEBUGGING"
    REFACTORING = "REFACTORING"
    RESEARCH = "RESEARCH"
    PLANNING = "PLANNING"
    DOCUMENTATION = "DOCUMENTATION"
    UNKNOWN = "UNKNOWN"

    @property
    def description(self) -> str:
        """Human-readable description of intent."""
        descriptions = {
            QueryIntent.CODING: "Writing new code or implementing features",
            QueryIntent.DEBUGGING: "Fixing bugs or resolving errors",
            QueryIntent.REFACTORING: "Restructuring or cleaning up code",
            QueryIntent.RESEARCH: "Understanding codebase or exploring",
            QueryIntent.PLANNING: "Architecture, design, or strategy",
            QueryIntent.DOCUMENTATION: "Writing documentation or comments",
            QueryIntent.UNKNOWN: "Unclear or mixed intent",
        }
        return descriptions[self]


@dataclass
class IntentMatch:
    """Result of intent classification."""

    intent: QueryIntent
    confidence: float  # 0-1, based on pattern match strength
    matched_patterns: List[str]  # Patterns that matched


class IntentClassifier:
    """
    Lightweight rule-based intent classifier.

    Uses regex patterns to avoid LLM calls (cost optimization).
    Patterns are matched case-insensitively.
    """

    # Pattern definitions for each intent
    # Pattern order matters - more specific patterns should be checked first
    PATTERNS: Dict[QueryIntent, List[str]] = {
        QueryIntent.DEBUGGING: [
            r'\b(fix|bug|error|issue|broken|failing|crash|crashes|exception)\b',
            r'\b(doesn\'t work|not working|won\'t|can\'t)\b',
            r'\b(stack trace|traceback|debug|diagnose)\b',
            r'\b(resolve|troubleshoot|investigate error)\b',
        ],
        QueryIntent.RESEARCH: [
            r'\b(how does|how do|where is|what is|what does)\b',
            r'\b(explain|understand|learn|explore|find out)\b',
            r'\b(what.{0,30}purpose|what.{0,30}mean)\b',
            r'\b(show me|tell me about|describe)\b',
        ],
        QueryIntent.REFACTORING: [
            r'\b(refactor|restructure|reorganize|cleanup)\b',
            r'\bclean\s+up\b',
            r'\b(extract.{0,20}(function|method|class))\b',
            r'\b(improve|optimize|enhance|polish)\b',
            r'\b(split|merge|consolidate|deduplicate)\b',
        ],
        QueryIntent.CODING: [
            r'\b(implement|add|create|write|build|make)\b',
            r'\b(new (feature|function|endpoint|component|class|method))\b',
            r'\b(develop|code|program)\b',
            r'\b(integrate|connect|hook up)\b',
        ],
        QueryIntent.PLANNING: [
            r'\b(plan|design|architect|strategy|approach)\b',
            r'\b(should we|should I|what if|consider)\b',
            r'\b(decide|decision|choose|evaluate|compare)\b',
            r'\b(roadmap|milestone|goal|objective)\b',
        ],
        QueryIntent.DOCUMENTATION: [
            r'\b(document|docs|documentation|readme|comment)\b',
            r'\b(write|update|add) .{0,20}(docstring|comment|documentation)\b',
            r'\b(explain in|describe in|document in)\b',
            r'\b(api docs|user guide|tutorial)\b',
        ],
    }

    def __init__(self):
        """Initialize classifier with compiled patterns."""
        # Compile patterns for performance
        self.compiled_patterns: Dict[QueryIntent, List[re.Pattern]] = {}
        for intent, patterns in self.PATTERNS.items():
            self.compiled_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def classify(self, query: str) -> IntentMatch:
        """
        Classify query intent using pattern matching.

        Args:
            query: User query string

        Returns:
            IntentMatch with detected intent and confidence
        """
        # Score each intent
        scores: Dict[QueryIntent, Tuple[int, List[str]]] = {
            intent: (0, []) for intent in QueryIntent if intent != QueryIntent.UNKNOWN
        }

        for intent, patterns in self.compiled_patterns.items():
            matched = []
            for pattern in patterns:
                if pattern.search(query):
                    scores[intent] = (scores[intent][0] + 1, scores[intent][1] + [pattern.pattern])
                    matched.append(pattern.pattern)

            if matched:
                scores[intent] = (scores[intent][0], matched)

        # Find best match
        if max(score[0] for score in scores.values()) == 0:
            # No matches - return UNKNOWN
            return IntentMatch(
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                matched_patterns=[],
            )

        best_intent = max(scores, key=lambda i: scores[i][0])
        match_count = scores[best_intent][0]
        matched_patterns = scores[best_intent][1]

        # Calculate confidence based on match count and pattern strength
        # More matches = higher confidence
        confidence = min(1.0, match_count * 0.25)

        return IntentMatch(
            intent=best_intent,
            confidence=confidence,
            matched_patterns=matched_patterns,
        )

    def classify_simple(self, query: str) -> QueryIntent:
        """
        Simplified classification returning only intent.

        Args:
            query: User query string

        Returns:
            QueryIntent enum
        """
        return self.classify(query).intent

    def get_intent_name(self, query: str) -> str:
        """
        Get intent as string (for use with HierarchyConfig).

        Args:
            query: User query string

        Returns:
            Intent name string (e.g., "DEBUGGING")
        """
        return self.classify_simple(query).value
