"""Tests for the web app's reused scientific engine and download formats."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solar_jv_engine import (
    _fit_core, estimate_all, parse_two_columns, raw_to_internal,
    solve_j_single, solve_j_two, suggest_regions, verify_physical_equation,
)
from web_exports import compact_parameter_table, origin_four_column_table


class TestSolarJV(unittest.TestCase):
    def test_known_single_diode_parameters(self):
        text = (ROOT / "example_solar_dark_jv_mAcm2.txt").read_text()
        df = parse_two_columns(text)
        vr, yr = df["V_raw"].to_numpy(float), df["Y_raw"].to_numpy(float)
        v, j, _ = raw_to_internal(vr, yr, "V + J (mA/cm^2)", 0.0256)
        regions, _ = suggest_regions(v, j, 300)
        estimates = estimate_all(v, j, 300, regions["rsh"], regions["diode"], regions["rs"])
        result = _fit_core(v, j, 300, estimates, "Balanced (asinh)", 4,
                           check_sensitivity=True)
        self.assertAlmostEqual(result["J0"] / 1e-10, 1, places=4)
        self.assertAlmostEqual(result["n"], 1.5, places=4)
        self.assertAlmostEqual(result["rs"], 3.0, places=4)
        self.assertAlmostEqual(result["rsh"] / 1e5, 1, places=4)
        self.assertLess(result["closure"], 1e-7)

        grid_v = np.linspace(np.min(v), np.max(v), 600)
        grid_j = solve_j_single(grid_v, result["J0"], result["n"],
                                result["rs"], result["rsh"], 300)
        origin = origin_four_column_table(vr, yr, grid_v, grid_j,
                                          "V + J (mA/cm^2)", 0.0256)
        self.assertEqual(origin.shape, (600, 4))
        np.testing.assert_allclose(origin.iloc[:len(vr), 0], vr)
        np.testing.assert_allclose(origin.iloc[:len(yr), 1], yr)
        self.assertEqual(origin.iloc[:, 2].notna().sum(), 600)
        self.assertEqual(compact_parameter_table(result)["Parameter"].tolist(),
                         ["J0", "n", "rs", "rsh"])

    def test_two_diode_constrained_fit_and_export(self):
        v = np.linspace(-0.2, 0.95, 125)
        truth = dict(J01=1e-11, n1=1.0, J02=3e-8, n2=2.0,
                     rs=2.5, rsh=50000.0)
        j = solve_j_two(v, **truth, T=300)
        regions, _ = suggest_regions(v, j, 300)
        estimates = estimate_all(v, j, 300, regions["rsh"], regions["diode"], regions["rs"])
        result = _fit_core(v, j, 300, estimates, "Balanced (asinh)", 5,
                           two=True, constrain_two=True, check_sensitivity=False)
        self.assertGreater(result["R2"], 0.999999)
        self.assertLess(result["closure"], 1e-7)
        self.assertEqual(result["n1"], 1.0)
        self.assertEqual(result["n2"], 2.0)
        self.assertEqual(compact_parameter_table(result)["Parameter"].tolist(),
                         ["J01", "n1", "J02", "n2", "rs", "rsh"])

    def test_acquisition_order_and_input_conversion(self):
        vr = np.array([0.4, 0.2, 0.0, -0.2, 0.2])
        ya = np.array([4e-6, 2e-6, 0.0, -2e-6, 2e-6])
        v, j, f = raw_to_internal(vr, ya, "V + I (A)", 0.02,
                                  flip_voltage=True, flip_current=True)
        np.testing.assert_array_equal(v, -vr)
        np.testing.assert_allclose(j, -ya / 0.02)
        self.assertTrue(f.all())


if __name__ == "__main__":
    unittest.main()
