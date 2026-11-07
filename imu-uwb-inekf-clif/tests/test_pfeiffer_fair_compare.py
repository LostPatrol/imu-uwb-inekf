from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ESTIMATION = PROJECT / "scripts" / "estimation"
sys.path.insert(0, str(ESTIMATION))

from pfeiffer_fair_compare import (  # noqa: E402
    PROTOCOLS,
    calibration_for_sequence,
    position_metrics,
)


class PfeifferFairProtocolTests(unittest.TestCase):
    def test_truth_free_protocols_disable_both_runtime_truth_paths(self) -> None:
        for name in ("no_oracle_raw", "no_oracle_loso"):
            protocol = PROTOCOLS[name]
            self.assertFalse(protocol.altitude_enabled)
            self.assertFalse(protocol.truth_dependent_range_sigma)

    def test_paper_protocol_documents_both_truth_paths(self) -> None:
        protocol = PROTOCOLS["paper_oracle"]
        self.assertTrue(protocol.altitude_enabled)
        self.assertTrue(protocol.truth_dependent_range_sigma)
        self.assertEqual(protocol.score_start_s, 1.0)

    def test_position_metrics_separate_xy_and_z_without_changing_3d(self) -> None:
        error = np.array([[3.0, 4.0, 12.0], [0.0, 0.0, 0.0]])
        metrics = position_metrics(error)
        self.assertAlmostEqual(metrics["position_rmse_m"], 13.0 / np.sqrt(2.0))
        self.assertAlmostEqual(metrics["position_xy_rmse_m"], 5.0 / np.sqrt(2.0))
        self.assertAlmostEqual(metrics["position_z_rmse_m"], 12.0 / np.sqrt(2.0))

    def test_loso_lookup_is_sequence_specific(self) -> None:
        sequence = "held_out.csv"
        table = pd.DataFrame(
            {
                "held_out_sequence": [sequence] * 8,
                "anchor_index": list(range(8)),
                "range_bias_m": np.arange(8) / 10.0,
                "range_sigma_m": np.arange(8) / 100.0 + 0.1,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.csv"
            table.to_csv(path, index=False)
            bias, sigma = calibration_for_sequence(path, sequence, "no_oracle_loso")
        np.testing.assert_allclose(bias, np.arange(8) / 10.0)
        np.testing.assert_allclose(sigma, np.arange(8) / 100.0 + 0.1)


if __name__ == "__main__":
    unittest.main()
