import unittest

import joblib
import pandas as pd

from app import render, validate
from model import DATA, FEATURES, MODEL, split


class PriceLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(MODEL)

    def test_splits_do_not_share_input_groups(self):
        df = pd.read_csv(DATA)
        sets = [set(pd.util.hash_pandas_object(df.iloc[i][FEATURES], index=False)) for i in split(df)]
        for i, rows in enumerate(sets):
            for other in sets[:i]:
                self.assertFalse(rows & other)

    def test_page_default_and_second_locality(self):
        status, html = render(self.bundle, {})
        self.assertEqual(status, 200)
        self.assertIn("Whitefield", html)
        self.assertIn("Historical prediction interval", html)
        status, other = render(self.bundle, {"locality": ["electronic city"], "sqft": ["1600"], "bhk": ["3"]})
        self.assertEqual(status, 200)
        self.assertIn("Electronic City", other)
        self.assertNotEqual(html, other)

    def test_invalid_input_is_escaped(self):
        status, html = render(self.bundle, {"sqft": ['"><script>alert(1)</script>']})
        self.assertEqual(status, 400)
        self.assertNotIn("<script>", html)

    def test_numeric_validation(self):
        base = dict(locality="whitefield", sqft="1000", bhk="2", bath="2")
        for key, value in [("bhk", "2.5"), ("bath", "NaN"), ("sqft", "250"), ("sqft", "inf")]:
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                validate({**base,key:value}, self.bundle[2])

    def test_quality_above_baseline(self):
        report = self.bundle[3]
        self.assertLess(report["test_metrics"][report["model"]]["mae_lakh"],
                        report["test_metrics"]["Global median"]["mae_lakh"])
        self.assertTrue(.72 <= report["test_interval_coverage"] <= .88)


if __name__ == "__main__":
    unittest.main()
