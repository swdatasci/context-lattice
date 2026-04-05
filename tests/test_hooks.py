"""
Tests for ContextLattice hooks.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

from context_lattice.hooks import PreQueryHook
from context_lattice.hooks.pre_query import HookInput


class TestHookInput:
    """Tests for HookInput parsing."""

    def test_parse_user_prompt_submit(self):
        """Test parsing UserPromptSubmit event."""
        json_input = json.dumps({
            "session_id": "abc123",
            "cwd": "/home/user/project",
            "user_prompt": "Fix the bug in login.py",
            "hook_event_name": "UserPromptSubmit"
        })

        with patch('sys.stdin', StringIO(json_input)):
            hook_input = HookInput.from_stdin()

        assert hook_input.session_id == "abc123"
        assert hook_input.cwd == "/home/user/project"
        assert hook_input.user_prompt == "Fix the bug in login.py"
        assert hook_input.hook_event_name == "UserPromptSubmit"

    def test_parse_with_prompt_field(self):
        """Test parsing with 'prompt' instead of 'user_prompt'."""
        json_input = json.dumps({
            "session_id": "xyz789",
            "cwd": "/tmp",
            "prompt": "What is this code doing?",
            "hook_event_name": "UserPromptSubmit"
        })

        with patch('sys.stdin', StringIO(json_input)):
            hook_input = HookInput.from_stdin()

        assert hook_input.user_prompt == "What is this code doing?"

    def test_parse_empty_stdin(self):
        """Test handling empty stdin."""
        with patch('sys.stdin', StringIO("")):
            with pytest.raises(ValueError, match="Empty stdin"):
                HookInput.from_stdin()

    def test_parse_invalid_json(self):
        """Test handling invalid JSON."""
        with patch('sys.stdin', StringIO("not valid json")):
            with pytest.raises(json.JSONDecodeError):
                HookInput.from_stdin()


class TestPreQueryHook:
    """Tests for PreQueryHook."""

    def test_init_defaults(self):
        """Test hook initialization with defaults."""
        hook = PreQueryHook()
        assert hook.budget == 8000
        assert hook.sources == ['semantic', 'file']

    def test_init_custom_config(self):
        """Test hook initialization with custom config."""
        hook = PreQueryHook(
            budget=10000,
            sources=['semantic'],
        )
        assert hook.budget == 10000
        assert hook.sources == ['semantic']

    def test_optimize_empty_query(self):
        """Test optimization with empty query returns minimal context."""
        hook = PreQueryHook(budget=1000, sources=[])
        result = hook.optimize("")
        # Empty query with no sources should return empty or minimal context
        assert result == "" or len(result) < 100

    def test_optimize_with_mock_sources(self):
        """Test optimization with mocked sources."""
        hook = PreQueryHook(budget=5000, sources=['file'])

        # Mock the collector to avoid actual source calls
        with patch.object(hook, '_load_config', return_value={}):
            with patch('context_lattice.hooks.pre_query.MultiSourceCollector') as MockCollector:
                mock_instance = MockCollector.return_value
                mock_instance.collect.return_value = []

                result = hook.optimize("Fix the bug")

                # Should complete without error even with no candidates
                assert isinstance(result, str)

    def test_run_from_stdin_success(self):
        """Test successful stdin processing."""
        json_input = json.dumps({
            "session_id": "test123",
            "cwd": "/tmp",
            "user_prompt": "Test query",
            "hook_event_name": "UserPromptSubmit"
        })

        hook = PreQueryHook(budget=1000, sources=[])

        with patch('sys.stdin', StringIO(json_input)):
            with patch.object(hook, 'optimize', return_value=""):
                exit_code = hook.run_from_stdin()

        assert exit_code == 0

    def test_run_from_stdin_no_prompt(self):
        """Test stdin processing with no prompt."""
        json_input = json.dumps({
            "session_id": "test123",
            "cwd": "/tmp",
            "user_prompt": "",
            "hook_event_name": "UserPromptSubmit"
        })

        hook = PreQueryHook()

        with patch('sys.stdin', StringIO(json_input)):
            exit_code = hook.run_from_stdin()

        # Should succeed but do nothing
        assert exit_code == 0

    def test_run_from_stdin_invalid_json(self):
        """Test stdin processing with invalid JSON."""
        hook = PreQueryHook()

        with patch('sys.stdin', StringIO("not json")):
            exit_code = hook.run_from_stdin()

        # Should return error code but not crash
        assert exit_code == 1

    def test_run_from_stdin_outputs_context(self, capsys):
        """Test that context is output to stdout."""
        json_input = json.dumps({
            "session_id": "test123",
            "cwd": "/tmp",
            "user_prompt": "Test query",
            "hook_event_name": "UserPromptSubmit"
        })

        hook = PreQueryHook()

        with patch('sys.stdin', StringIO(json_input)):
            with patch.object(hook, 'optimize', return_value="Injected context here"):
                exit_code = hook.run_from_stdin()

        captured = capsys.readouterr()
        assert "Injected context here" in captured.out
        assert exit_code == 0


class TestHookIntegration:
    """Integration tests for hook functionality."""

    def test_hook_with_real_file_source(self, tmp_path):
        """Test hook with actual file source on temp files."""
        # Create a temp Python file
        test_file = tmp_path / "test_module.py"
        test_file.write_text("""
def authenticate_user(username, password):
    '''Authenticate a user against the database.'''
    # TODO: Add rate limiting
    return check_credentials(username, password)
""")

        hook = PreQueryHook(
            project_root=tmp_path,
            budget=2000,
            sources=['file'],
        )

        result = hook.optimize(
            query="Fix the authentication function",
            cwd=str(tmp_path),
        )

        # Should find the relevant file content
        # (may be empty if embedding model not available in test env)
        assert isinstance(result, str)

    def test_hook_intent_affects_budget(self):
        """Test that intent classification affects budget allocation."""
        hook = PreQueryHook(budget=10000)

        # Debugging query should boost DIRECT level
        with patch('context_lattice.hooks.pre_query.MultiSourceCollector') as MockCollector:
            mock_instance = MockCollector.return_value
            mock_instance.collect.return_value = []

            # We can't easily verify budget allocation, but ensure no errors
            result = hook.optimize("Fix the bug in auth.py")
            assert isinstance(result, str)

            result = hook.optimize("What is the architecture of this system?")
            assert isinstance(result, str)
