"""End-to-end local checks of CLI output using a clearly marked test client."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from test_comparison import VALID_RESPONSE, ScriptedClient

import compare_prompts
import main


class CLITests(unittest.TestCase):
    def test_demo_uses_selected_prompt_and_saves_three_results(self) -> None:
        client = ScriptedClient([VALID_RESPONSE] * 3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "demo.json"
            argv = [
                "main.py",
                "--demo",
                "--prompt-variant",
                "example",
                "--output",
                str(output),
            ]
            stdout = io.StringIO()
            with (
                patch.object(main, "LLMClient", return_value=client),
                patch("sys.argv", argv),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main.main(), 0)
            results = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(results), 3)
            self.assertEqual(results, json.loads(stdout.getvalue()))
            self.assertTrue(
                all("Example input:" in system for system, _ in client.calls)
            )

    def test_comparison_cli_saves_all_nine_attempts(self) -> None:
        client = ScriptedClient([VALID_RESPONSE] * 9)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            argv = [
                "compare_prompts.py",
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
            with (
                patch.object(
                    compare_prompts, "LLMClient", return_value=client
                ),
                patch("sys.argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(compare_prompts.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]), 9)
            self.assertEqual(report["model"], "test-fixture")


if __name__ == "__main__":
    unittest.main()
