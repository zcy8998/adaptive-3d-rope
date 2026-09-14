import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from util.checkpoint import load_checkpoint


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_namespace_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"args": argparse.Namespace(seed=42), "model": {"x": torch.ones(1)}}, path)
            restored = load_checkpoint(path)
        self.assertEqual(restored["args"].seed, 42)
        self.assertEqual(float(restored["model"]["x"].item()), 1.0)


if __name__ == "__main__":
    unittest.main()
