"""
Context assembler: Format optimized context for query augmentation.

Combines context from all hierarchy levels with clear headers and formatting.
"""

from typing import List, Dict
from dataclasses import dataclass

from .hierarchy import HierarchyLevel
from .node import ContextNode
from .budget import ContextBudget


@dataclass
class AssembledContext:
    """Assembled context ready for query augmentation."""

    text: str
    total_tokens: int
    per_level_tokens: Dict[HierarchyLevel, int]
    per_level_count: Dict[HierarchyLevel, int]
    efficiency: float  # Used tokens / allocated budget

    def to_dict(self) -> Dict[str, any]:
        """Serialize for logging."""
        return {
            "total_tokens": self.total_tokens,
            "efficiency": round(self.efficiency, 3),
            "per_level_tokens": {
                level.name: tokens for level, tokens in self.per_level_tokens.items()
            },
            "per_level_count": {
                level.name: count for level, count in self.per_level_count.items()
            },
        }


class ContextAssembler:
    """
    Assemble context from hierarchy pools into formatted text.

    Formats context with clear headers for each level:
    - STRUCTURAL (Always Included)
    - DIRECT (Query-Matched)
    - IMPLIED (Related)
    - BACKGROUND (Architectural)
    """

    def assemble(
        self,
        selected: Dict[HierarchyLevel, List[ContextNode]],
        budget: ContextBudget,
    ) -> AssembledContext:
        """
        Assemble context from selected nodes.

        Args:
            selected: Dictionary mapping level to selected nodes
            budget: Original budget allocation

        Returns:
            AssembledContext with formatted text and metadata
        """
        sections = []
        total_tokens = 0
        per_level_tokens = {}
        per_level_count = {}

        # Process each level in order
        for level in [
            HierarchyLevel.STRUCTURAL,
            HierarchyLevel.DIRECT,
            HierarchyLevel.IMPLIED,
            HierarchyLevel.BACKGROUND,
        ]:
            nodes = selected.get(level, [])
            if not nodes:
                per_level_tokens[level] = 0
                per_level_count[level] = 0
                continue

            # Create section
            section = self._format_section(level, nodes)
            sections.append(section)

            # Track stats
            level_tokens = sum(n.tokens for n in nodes)
            total_tokens += level_tokens
            per_level_tokens[level] = level_tokens
            per_level_count[level] = len(nodes)

        # Combine sections
        text = "\n\n".join(sections)

        # Calculate efficiency
        total_budget = budget.total_allocated
        efficiency = total_tokens / total_budget if total_budget > 0 else 0.0

        return AssembledContext(
            text=text,
            total_tokens=total_tokens,
            per_level_tokens=per_level_tokens,
            per_level_count=per_level_count,
            efficiency=efficiency,
        )

    def _format_section(self, level: HierarchyLevel, nodes: List[ContextNode]) -> str:
        """
        Format a single hierarchy level section.

        Args:
            level: Hierarchy level
            nodes: Nodes in this level

        Returns:
            Formatted section text
        """
        # Header
        header = self._get_section_header(level)
        lines = [header, ""]

        # Format each node
        for node in nodes:
            # Add file path if available
            if file_path := node.metadata.get('file_path'):
                lines.append(f"## {file_path}")

            # Add entity name if available
            if entity_name := node.metadata.get('entity_name'):
                lines.append(f"### {entity_name}")

            # Add content
            lines.append(node.content)
            lines.append("")  # Blank line between nodes

        return "\n".join(lines)

    def _get_section_header(self, level: HierarchyLevel) -> str:
        """Get formatted header for hierarchy level."""
        headers = {
            HierarchyLevel.STRUCTURAL: "# STRUCTURAL CONTEXT (Always Included)",
            HierarchyLevel.DIRECT: "# DIRECT CONTEXT (Query-Matched)",
            HierarchyLevel.IMPLIED: "# IMPLIED CONTEXT (Related)",
            HierarchyLevel.BACKGROUND: "# BACKGROUND CONTEXT (Architectural)",
        }
        return headers[level]

    def assemble_minimal(self, nodes: List[ContextNode]) -> str:
        """
        Assemble minimal context without headers (for embedding).

        Args:
            nodes: List of context nodes

        Returns:
            Concatenated content
        """
        return "\n\n".join(node.content for node in nodes)
