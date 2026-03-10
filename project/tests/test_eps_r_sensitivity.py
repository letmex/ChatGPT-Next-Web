import math
import unittest

from project.eps_r_sensitivity import run_eps_r_sensitivity


class EpsRSensitivityTest(unittest.TestCase):
    def test_records_metrics_for_requested_eps_values(self):
        rows = run_eps_r_sensitivity(eps_values=[1e-4, 1e-5, 1e-6], n_points=64, n_steps=10)
        self.assertEqual([r["eps_r"] for r in rows], [1e-4, 1e-5, 1e-6])

        for row in rows:
            self.assertGreater(row["HI_peak"], 0.0)
            self.assertGreater(row["HII_peak"], 0.0)
            self.assertTrue(math.isfinite(row["first_crack_t"]))

    def test_comsol_aligned_value_tracks_small_epsilon_behavior(self):
        rows = run_eps_r_sensitivity(eps_values=[1e-4, 1e-5, 1e-6], n_points=64, n_steps=10)
        by_eps = {r["eps_r"]: r for r in rows}

        diff_1e5_1e6 = abs(by_eps[1e-5]["HII_peak"] - by_eps[1e-6]["HII_peak"])
        diff_1e4_1e5 = abs(by_eps[1e-4]["HII_peak"] - by_eps[1e-5]["HII_peak"])
        self.assertLess(diff_1e5_1e6, diff_1e4_1e5)


if __name__ == "__main__":
    unittest.main()
