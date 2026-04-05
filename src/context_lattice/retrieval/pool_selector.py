"""
Pool selection: Hierarchy-based context filtering.

Assigns context nodes to hierarchy levels (pools) based on structural rules,
not just semantic similarity. This ensures critical context is always included.
"""

from typing import List, Dict, Set
import re
from pathlib import Path

from ..core.hierarchy import HierarchyLevel
from ..core.node import ContextNode


class PoolSelector:
    """
    Assign context nodes to hierarchy pools based on structural rules.

    Pools are levels in the semantic hierarchy:
    - STRUCTURAL: Always include (user prefs, project CLAUDE.md, current task)
    - DIRECT: Explicitly mentioned in query (files, entities, recent turns)
    - IMPLIED: Same module/directory as direct items
    - BACKGROUND: Everything else (ranked by similarity)
    """

    def __init__(self, project_root: Path = None):
        """
        Initialize pool selector.

        Args:
            project_root: Root directory of current project
        """
        self.project_root = project_root or Path.cwd()

    def assign_pools(
        self,
        candidates: List[ContextNode],
        query: str,
        current_file: str = None,
    ) -> Dict[HierarchyLevel, List[ContextNode]]:
        """
        Assign candidates to hierarchy pools.

        Args:
            candidates: All context nodes available
            query: User query string
            current_file: Currently open/focused file (if any)

        Returns:
            Dictionary mapping HierarchyLevel to list of nodes
        """
        pools = {
            HierarchyLevel.STRUCTURAL: [],
            HierarchyLevel.DIRECT: [],
            HierarchyLevel.IMPLIED: [],
            HierarchyLevel.BACKGROUND: [],
        }

        # Extract mentioned files/entities from query
        mentioned_files = self._extract_file_mentions(query)
        mentioned_entities = self._extract_entity_mentions(query)

        # Get direct file directories for IMPLIED level
        direct_dirs = set()

        for node in candidates:
            # Check STRUCTURAL level (always include certain types)
            if self._is_structural(node):
                pools[HierarchyLevel.STRUCTURAL].append(node)
                continue

            # Check DIRECT level (mentioned in query or current file)
            if self._is_direct(node, query, mentioned_files, mentioned_entities, current_file):
                pools[HierarchyLevel.DIRECT].append(node)
                # Track directory for IMPLIED level
                if file_path := node.metadata.get('file_path'):
                    direct_dirs.add(str(Path(file_path).parent))
                continue

            # Check IMPLIED level (same module as direct)
            if self._is_implied(node, direct_dirs, pools[HierarchyLevel.DIRECT]):
                pools[HierarchyLevel.IMPLIED].append(node)
                continue

            # Otherwise, BACKGROUND level
            pools[HierarchyLevel.BACKGROUND].append(node)

        return pools

    def _is_structural(self, node: ContextNode) -> bool:
        """
        Check if node is STRUCTURAL (always include).

        STRUCTURAL includes:
        - User preferences/corrections
        - Project CLAUDE.md
        - Active task definition
        - System conventions
        """
        # Check metadata tags
        if node.metadata.get('type') in ['user_preference', 'user_correction']:
            return True

        if node.metadata.get('type') == 'project_context':
            return True

        # Check file path
        if file_path := node.metadata.get('file_path'):
            file_name = Path(file_path).name.lower()
            if file_name in ['claude.md', 'user_preferences.yaml', 'user_prefs.yaml']:
                return True

        # Check if marked as structural
        if node.metadata.get('structural', False):
            return True

        return False

    def _is_direct(
        self,
        node: ContextNode,
        query: str,
        mentioned_files: Set[str],
        mentioned_entities: Set[str],
        current_file: str = None,
    ) -> bool:
        """
        Check if node is DIRECT (explicitly mentioned or current).

        DIRECT includes:
        - Files mentioned in query
        - Entities mentioned in query (functions, classes)
        - Current/focused file
        - Recent conversation turns
        """
        # Check if current file
        if current_file:
            if file_path := node.metadata.get('file_path'):
                if Path(file_path).resolve() == Path(current_file).resolve():
                    return True

        # Check if file mentioned in query
        if file_path := node.metadata.get('file_path'):
            file_name = Path(file_path).name
            if file_name in mentioned_files:
                return True
            # Also check full path
            if any(mention in str(file_path) for mention in mentioned_files):
                return True

        # Check if entity mentioned in query
        if entity_name := node.metadata.get('entity_name'):
            if entity_name in mentioned_entities:
                return True

        # Check if recent conversation
        if node.metadata.get('type') == 'conversation':
            # Recent conversation is DIRECT
            return True

        # Check if explicitly marked as direct
        if node.metadata.get('direct', False):
            return True

        return False

    def _is_implied(
        self,
        node: ContextNode,
        direct_dirs: Set[str],
        direct_nodes: List[ContextNode],
    ) -> bool:
        """
        Check if node is IMPLIED (related to direct).

        IMPLIED includes:
        - Files in same directory as DIRECT files
        - Type definitions used by DIRECT code
        - Test files for DIRECT code
        """
        # Check if in same directory as direct files
        if file_path := node.metadata.get('file_path'):
            node_dir = str(Path(file_path).parent)
            if node_dir in direct_dirs:
                return True

        # Check if test file for direct code
        if file_path := node.metadata.get('file_path'):
            file_name = Path(file_path).name
            if 'test' in file_name.lower():
                # Check if tests for any direct file
                for direct_node in direct_nodes:
                    if direct_file := direct_node.metadata.get('file_path'):
                        direct_name = Path(direct_file).stem
                        if direct_name in file_name:
                            return True

        # Check if type definition used by direct code
        if node.metadata.get('type') == 'type_definition':
            # Could implement AST analysis here to check imports
            # For MVP, just check if in same module
            if file_path := node.metadata.get('file_path'):
                node_dir = str(Path(file_path).parent)
                if node_dir in direct_dirs:
                    return True

        return False

    def _extract_file_mentions(self, query: str) -> Set[str]:
        """
        Extract file paths/names mentioned in query.

        Returns:
            Set of file names/paths mentioned
        """
        mentions = set()

        # Pattern: filename.ext or path/to/filename.ext
        file_pattern = r'\b([\w/-]+\.(py|js|ts|tsx|jsx|md|yaml|yml|json|txt|rs|go|java|cpp|c|h|hpp))\b'
        for match in re.finditer(file_pattern, query, re.IGNORECASE):
            mentions.add(match.group(1))
            # Also add just the filename
            mentions.add(Path(match.group(1)).name)

        return mentions

    def _extract_entity_mentions(self, query: str) -> Set[str]:
        """
        Extract code entities (functions, classes) mentioned in query.

        Returns:
            Set of entity names mentioned
        """
        mentions = set()

        # Pattern: PascalCase or snake_case identifiers
        # Look for them in context like "fix the login_handler function"
        entity_pattern = r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*|[a-z_]+[a-z0-9_]*)\b'

        # Only extract if followed by keywords like "function", "class", "method"
        context_pattern = r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*|[a-z_]+[a-z0-9_]*)\s+(function|class|method|handler|component|service|module)\b'

        for match in re.finditer(context_pattern, query, re.IGNORECASE):
            mentions.add(match.group(1))

        return mentions
