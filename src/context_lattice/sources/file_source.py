"""
File source: Read files directly from filesystem.

Extracts functions, classes, and content from local files.
"""

from typing import List, Optional, Set
import logging
from pathlib import Path
from datetime import datetime
import re

from sentence_transformers import SentenceTransformer
import numpy as np

from ..core.node import ContextNode
from ..core.hierarchy import HierarchyLevel

logger = logging.getLogger(__name__)


class FileSource:
    """
    Fetch context from local files.

    Reads files, extracts entities (functions, classes), and creates ContextNodes.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        file_types: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ):
        """
        Initialize file source.

        Args:
            model_name: Sentence transformer model for embeddings
            file_types: File extensions to include (default: [".py", ".ts", ".md", ".yaml"])
            exclude_dirs: Directories to exclude (default: [".git", "node_modules", ".venv"])
        """
        self.model = SentenceTransformer(model_name)

        self.file_types = file_types or [".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".yaml", ".yml"]
        self.exclude_dirs = exclude_dirs or [".git", "node_modules", ".venv", "dist", "build", "__pycache__"]

    def fetch(
        self,
        query: str,
        project_root: Path,
        current_file: Optional[str] = None,
        max_files: int = 20,
    ) -> List[ContextNode]:
        """
        Fetch context nodes from local files.

        Args:
            query: Search query (used to prioritize relevant files)
            project_root: Root directory of project
            current_file: Currently open file (gets highest priority)
            max_files: Maximum files to process

        Returns:
            List of ContextNode from files
        """
        try:
            # Find relevant files
            files = self._find_relevant_files(query, project_root, current_file, max_files)

            # Process files into nodes
            nodes = []
            for file_path in files:
                file_nodes = self._process_file(file_path, query)
                nodes.extend(file_nodes)

            logger.info(f"File source returned {len(nodes)} nodes from {len(files)} files")
            return nodes

        except Exception as e:
            logger.error(f"File source fetch failed: {e}")
            return []

    def _find_relevant_files(
        self,
        query: str,
        project_root: Path,
        current_file: Optional[str],
        max_files: int,
    ) -> List[Path]:
        """
        Find files relevant to query.

        Args:
            query: Search query
            project_root: Project root directory
            current_file: Current file path
            max_files: Maximum files

        Returns:
            List of file paths, sorted by relevance
        """
        files = []

        # Extract file mentions from query
        mentioned_files = self._extract_file_mentions(query)

        # Collect all eligible files
        for file_path in project_root.rglob("*"):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Skip if excluded directory
            if any(exc in file_path.parts for exc in self.exclude_dirs):
                continue

            # Skip if wrong file type
            if file_path.suffix not in self.file_types:
                continue

            files.append(file_path)

        # Score and sort files by relevance
        scored_files = []
        for file_path in files:
            score = self._score_file_relevance(file_path, query, mentioned_files, current_file)
            scored_files.append((score, file_path))

        # Sort by score (descending) and take top N
        scored_files.sort(reverse=True, key=lambda x: x[0])
        top_files = [f[1] for f in scored_files[:max_files]]

        return top_files

    def _score_file_relevance(
        self,
        file_path: Path,
        query: str,
        mentioned_files: Set[str],
        current_file: Optional[str],
    ) -> float:
        """
        Score file relevance to query.

        Args:
            file_path: File path
            query: Query string
            mentioned_files: Files mentioned in query
            current_file: Currently open file

        Returns:
            Relevance score (higher = more relevant)
        """
        score = 0.0

        # Current file gets highest score
        if current_file and Path(current_file).resolve() == file_path.resolve():
            score += 10.0

        # File mentioned in query
        if file_path.name in mentioned_files:
            score += 5.0

        # File path contains query keywords
        query_words = query.lower().split()
        file_str = str(file_path).lower()
        for word in query_words:
            if word in file_str:
                score += 1.0

        # Recency (newer files slightly preferred)
        try:
            age_days = (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).days
            recency_score = max(0, 1.0 - (age_days / 365))  # Decay over 1 year
            score += recency_score
        except:
            pass

        return score

    def _process_file(self, file_path: Path, query: str) -> List[ContextNode]:
        """
        Process a file into ContextNodes.

        Args:
            file_path: Path to file
            query: Search query

        Returns:
            List of ContextNode from file
        """
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')

            # Determine hierarchy level
            level = self._determine_level(file_path)

            # For code files, extract entities
            if file_path.suffix in ['.py', '.ts', '.tsx', '.js', '.jsx']:
                return self._extract_entities(file_path, content, level, query)

            # For markdown/docs, use whole file or chunks
            else:
                return self._chunk_document(file_path, content, level, query)

        except Exception as e:
            logger.warning(f"Failed to process file {file_path}: {e}")
            return []

    def _determine_level(self, file_path: Path) -> HierarchyLevel:
        """Determine hierarchy level based on file path."""
        file_name = file_path.name.lower()

        # STRUCTURAL: Project config, user prefs
        if file_name in ['claude.md', 'user_preferences.yaml', 'user_prefs.yaml', '.clinerules']:
            return HierarchyLevel.STRUCTURAL

        # DIRECT: Everything else starts as DIRECT (pool selector will refine)
        return HierarchyLevel.DIRECT

    def _extract_entities(
        self,
        file_path: Path,
        content: str,
        level: HierarchyLevel,
        query: str,
    ) -> List[ContextNode]:
        """
        Extract functions/classes from code files.

        Args:
            file_path: File path
            content: File content
            level: Hierarchy level
            query: Query (for embedding context)

        Returns:
            List of ContextNode (one per entity + one for whole file)
        """
        nodes = []

        # Extract functions (simple regex - can be improved with AST)
        if file_path.suffix == '.py':
            pattern = r'^\s*(def|class)\s+(\w+)'
        else:  # TypeScript/JavaScript
            pattern = r'^\s*(function|class|const|export\s+function|export\s+class)\s+(\w+)'

        matches = re.finditer(pattern, content, re.MULTILINE)

        for match in matches:
            entity_type = match.group(1)
            entity_name = match.group(2)

            # Get context around entity (simplified - could use AST)
            start = match.start()
            # Find end (next entity or end of file)
            next_match = re.search(pattern, content[start+len(match.group(0)):], re.MULTILINE)
            if next_match:
                end = start + len(match.group(0)) + next_match.start()
            else:
                end = len(content)

            entity_content = content[start:end].strip()

            # Create node for entity
            node_id = f"file_{file_path.stem}_{entity_name}"
            tokens = len(entity_content) // 4 + 1

            # Embed entity
            embedding = self.model.encode(entity_content)

            metadata = {
                "file_path": str(file_path),
                "entity_name": entity_name,
                "entity_type": entity_type,
                "source": "file",
            }

            nodes.append(ContextNode(
                id=node_id,
                content=entity_content,
                tokens=tokens,
                level=level,
                embedding=embedding,
                metadata=metadata,
                timestamp=datetime.fromtimestamp(file_path.stat().st_mtime),
            ))

        # Also create node for whole file (useful for STRUCTURAL files)
        if level == HierarchyLevel.STRUCTURAL or not nodes:
            whole_file_node = self._create_whole_file_node(file_path, content, level)
            nodes.insert(0, whole_file_node)

        return nodes

    def _chunk_document(
        self,
        file_path: Path,
        content: str,
        level: HierarchyLevel,
        query: str,
    ) -> List[ContextNode]:
        """
        Chunk markdown/yaml documents.

        Args:
            file_path: File path
            content: File content
            level: Hierarchy level
            query: Query

        Returns:
            List of ContextNode (chunked)
        """
        # For now, return whole file as single node
        # Can be improved with header-based chunking for markdown
        return [self._create_whole_file_node(file_path, content, level)]

    def _create_whole_file_node(
        self,
        file_path: Path,
        content: str,
        level: HierarchyLevel,
    ) -> ContextNode:
        """Create node for whole file."""
        node_id = f"file_{file_path.stem}_full"
        tokens = len(content) // 4 + 1

        # Embed file
        embedding = self.model.encode(content)

        metadata = {
            "file_path": str(file_path),
            "source": "file",
            "full_file": True,
        }

        return ContextNode(
            id=node_id,
            content=content,
            tokens=tokens,
            level=level,
            embedding=embedding,
            metadata=metadata,
            timestamp=datetime.fromtimestamp(file_path.stat().st_mtime),
        )

    def _extract_file_mentions(self, query: str) -> Set[str]:
        """Extract file names mentioned in query."""
        mentions = set()

        # Pattern: filename.ext
        file_pattern = r'\b([\w/-]+\.(py|js|ts|tsx|jsx|md|yaml|yml|json|txt))\b'
        for match in re.finditer(file_pattern, query, re.IGNORECASE):
            mentions.add(match.group(1))
            mentions.add(Path(match.group(1)).name)  # Also add just filename

        return mentions
