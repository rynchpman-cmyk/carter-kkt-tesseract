import tempfile
import unittest
from pathlib import Path

import numpy as np

from tesseract_nr.io import load_checkpoint, save_checkpoint


class IOTests(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            save_checkpoint(path, metadata={"branch": "test"}, field=np.arange(5), time=1.25)
            arrays, metadata = load_checkpoint(path)
            np.testing.assert_array_equal(arrays["field"], np.arange(5))
            self.assertEqual(float(arrays["time"]), 1.25)
            self.assertEqual(metadata["branch"], "test")


if __name__ == "__main__":
    unittest.main()
