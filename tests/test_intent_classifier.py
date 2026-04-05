"""Tests for intent classifier."""

import pytest
from context_lattice.retrieval import QueryIntent, IntentClassifier


def test_debugging_intent():
    """Test debugging intent classification."""
    classifier = IntentClassifier()

    queries = [
        "Fix the authentication bug in login.py",
        "Error in user registration",
        "This function crashes when called",
    ]

    for query in queries:
        result = classifier.classify(query)
        assert result.intent == QueryIntent.DEBUGGING
        assert result.confidence > 0


def test_coding_intent():
    """Test coding intent classification."""
    classifier = IntentClassifier()

    queries = [
        "Implement a new user registration feature",
        "Add a logout endpoint to the API",
        "Create a password validation function",
    ]

    for query in queries:
        result = classifier.classify(query)
        assert result.intent == QueryIntent.CODING


def test_research_intent():
    """Test research intent classification."""
    classifier = IntentClassifier()

    queries = [
        "How does the authentication system work?",
        "Where is the user model defined?",
        "What is the purpose of this function?",
    ]

    for query in queries:
        result = classifier.classify(query)
        assert result.intent == QueryIntent.RESEARCH


def test_refactoring_intent():
    """Test refactoring intent classification."""
    classifier = IntentClassifier()

    queries = [
        "Refactor the authentication module",
        "Restructure the user service",
        "Extract this function from the class",
    ]

    for query in queries:
        result = classifier.classify(query)
        assert result.intent == QueryIntent.REFACTORING


def test_unknown_intent():
    """Test unknown intent classification."""
    classifier = IntentClassifier()

    queries = [
        "Hello",
        "What's the weather?",
        "Random text without patterns",
    ]

    for query in queries:
        result = classifier.classify(query)
        assert result.intent == QueryIntent.UNKNOWN
        assert result.confidence == 0.0


def test_intent_name_extraction():
    """Test getting intent name as string."""
    classifier = IntentClassifier()

    query = "Fix the bug in login"
    intent_name = classifier.get_intent_name(query)
    assert intent_name == "DEBUGGING"
    assert isinstance(intent_name, str)
