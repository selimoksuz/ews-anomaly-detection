import json
import tempfile
import unittest
from pathlib import Path

from engine.notebook_builder import build_simulation_notebook


class NotebookBuilderTests(unittest.TestCase):
    def test_build_simulation_notebook_creates_expected_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sim.ipynb"
            result = build_simulation_notebook(output_path)

            self.assertEqual(result, output_path)
            notebook = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(notebook["cells"]), 10)

            sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
            self.assertIn("run_batch", sources)
            self.assertIn("full_effects", sources)
            self.assertEqual(notebook["nbformat"], 4)


if __name__ == "__main__":
    unittest.main()
