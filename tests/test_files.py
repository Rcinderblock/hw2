"""Failed output saving must leave existing files intact."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import validate_output_path, write_json_output


class FileTests(unittest.TestCase):
    def test_failed_replace_preserves_old_output_and_cleans_temporary_file(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            output.write_text("old result", encoding="utf-8")
            with patch.object(
                Path, "replace", side_effect=OSError("Unavailable")
            ):
                with self.assertRaises(OSError):
                    write_json_output(output, '{"new": "result"}')
            self.assertEqual(output.read_text(encoding="utf-8"), "old result")
            self.assertEqual(list(root.iterdir()), [output])

    def test_input_output_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("original", encoding="utf-8")
            alias = root / "alias.txt"
            alias.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "входной файл"):
                validate_output_path(alias, [source])
            self.assertEqual(source.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
