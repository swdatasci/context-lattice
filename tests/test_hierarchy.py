"""Tests for hierarchy definitions."""

import pytest
from context_lattice.core import HierarchyLevel, HierarchyConfig


def test_hierarchy_levels():
    """Test hierarchy level definitions."""
    assert HierarchyLevel.STRUCTURAL.value == 0
    assert HierarchyLevel.DIRECT.value == 1
    assert HierarchyLevel.IMPLIED.value == 2
    assert HierarchyLevel.BACKGROUND.value == 3


def test_hierarchy_descriptions():
    """Test hierarchy level descriptions."""
    assert "user preferences" in HierarchyLevel.STRUCTURAL.description.lower()
    assert "query-matched" in HierarchyLevel.DIRECT.description.lower()
    assert "semantically related" in HierarchyLevel.IMPLIED.description.lower()
    assert "architecture" in HierarchyLevel.BACKGROUND.description.lower()


def test_default_budget_percentages():
    """Test default budget allocations."""
    total = sum(level.default_budget_pct for level in HierarchyLevel)
    assert abs(total - 1.0) < 0.01  # Allow small floating point error


def test_hierarchy_config_validation():
    """Test hierarchy config validation."""
    config = HierarchyConfig()
    assert config.validate()

    # Test invalid config
    with pytest.raises(ValueError):
        bad_config = HierarchyConfig(
            structural_pct=0.5,
            direct_pct=0.5,
            implied_pct=0.5,  # Sum > 1.0
            background_pct=0.5,
        )
        bad_config.validate()


def test_budget_allocation():
    """Test budget allocation with intent."""
    config = HierarchyConfig()

    # Test default allocation
    allocation = config.get_budget_allocation(10000, intent="UNKNOWN")
    assert sum(allocation.values()) == 10000

    # Test debugging intent (should boost DIRECT)
    debug_allocation = config.get_budget_allocation(10000, intent="DEBUGGING")
    normal_allocation = config.get_budget_allocation(10000, intent="UNKNOWN")

    assert debug_allocation[HierarchyLevel.DIRECT] > normal_allocation[HierarchyLevel.DIRECT]

    # Test research intent (should boost BACKGROUND)
    research_allocation = config.get_budget_allocation(10000, intent="RESEARCH")
    assert research_allocation[HierarchyLevel.BACKGROUND] > normal_allocation[HierarchyLevel.BACKGROUND]
