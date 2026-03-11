import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astock_prob.modeling.constraints import enforce_monotonic_touch_probabilities


class ConstraintTests(unittest.TestCase):
    def test_monotonic_touch_probabilities(self) -> None:
        raw = {
            "touch_up_5": 0.60,
            "touch_up_10": 0.65,
            "touch_up_15": 0.45,
        }
        adjusted = enforce_monotonic_touch_probabilities(raw, [0.05, 0.1, 0.15], "up")
        self.assertGreaterEqual(adjusted["touch_up_5"], adjusted["touch_up_10"])
        self.assertGreaterEqual(adjusted["touch_up_10"], adjusted["touch_up_15"])


if __name__ == "__main__":
    unittest.main()
