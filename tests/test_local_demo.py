"""Reject invalid requests before loading or downloading a local model."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from httpx import ConnectError

import local_demo
from local_client import LocalLLMClient


class LocalDemoTests(unittest.TestCase):
    def test_download_connection_error_is_readable_and_does_not_leak_details(
        self,
    ):
        fake_runtime = {
            "mlx": Mock(),
            "mlx.core": Mock(),
            "mlx_lm": Mock(
                load=Mock(side_effect=ConnectError("private server detail"))
            ),
            "mlx_lm.sample_utils": Mock(),
        }
        with patch.dict("sys.modules", fake_runtime):
            with self.assertRaisesRegex(ValueError, "ConnectError") as caught:
                LocalLLMClient()
        self.assertNotIn("private server detail", str(caught.exception))

    def test_invalid_samples_and_source_overwrite_do_not_load_a_model(self):
        for options in (
            ["--samples", "missing"],
            ["--samples", "01_meeting", "01_meeting"],
            [
                "--output",
                str(local_demo.PROJECT_DIR / "sample_inputs/01_meeting.txt"),
            ],
        ):
            with (
                self.subTest(options=options),
                patch("sys.argv", ["local_demo.py", *options]),
                patch.object(local_demo, "LocalLLMClient") as client,
                self.assertLogs(level="ERROR"),
            ):
                self.assertEqual(local_demo.main(), 1)
                client.assert_not_called()

    def test_missing_optional_dependency_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("previous result", encoding="utf-8")
            with (
                patch("sys.argv", ["local_demo.py", "--output", str(output)]),
                patch.dict("sys.modules", {"mlx": None}),
                self.assertLogs(level="ERROR") as captured,
            ):
                self.assertEqual(local_demo.main(), 1)
            self.assertIn("requirements-local.txt", "\n".join(captured.output))
            self.assertEqual(output.read_text(), "previous result")


if __name__ == "__main__":
    unittest.main()
