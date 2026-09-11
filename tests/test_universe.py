"""Offline regression tests: python -m unittest discover -s tests -v."""
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch
from urllib.error import HTTPError

import pandas as pd

from scan import universe


NOW = dt.datetime(2026, 9, 11, 0, 0, tzinfo=dt.timezone.utc)


class FrozenDateTime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


def listing():
    return pd.DataFrame([
        {"Code": f"{i:05d}0", "Name": f"Stock {i}",
         "Market": "KOSPI" if i < 500 else "KOSDAQ",
         "Marcap": (1200 - i) * 1000000, "Close": 5000}
        for i in range(1200)
    ])


class UniverseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.cache = Path(temp.name) / "data" / "universe_cache.json"
        self.df = listing()
        for target, value in (("CACHE_PATH", self.cache), ("dt.datetime", FrozenDateTime)):
            patcher = patch(f"scan.universe.{target}", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fetch = self.start_patch("fdr.StockListing")
        self.sleep = self.start_patch("time.sleep")
        self.log = self.start_patch("LOG.warning")
        self.fetch.return_value = self.df

    def start_patch(self, target):
        patcher = patch(f"scan.universe.{target}")
        self.addCleanup(patcher.stop)
        return patcher.start()

    def write_cache(self, age=dt.timedelta(days=1), df=None):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1, "fetched_at": (NOW - age).isoformat(),
            "rows": (self.df if df is None else df).to_dict(orient="records"),
        }
        self.cache.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fail_fetch(self):
        self.fetch.side_effect = HTTPError("https://example.test/listing", 404, "Not Found", {}, None)

    def test_success_on_each_attempt_saves_cache_and_stops_retrying(self):
        for attempt in range(1, 6):
            with self.subTest(attempt=attempt):
                self.fetch.reset_mock()
                self.sleep.reset_mock()
                self.fetch.side_effect = [OSError("HTTP 404")] * (attempt - 1) + [self.df]
                actual = universe._get_listing()
                pd.testing.assert_frame_equal(actual, self.df)
                self.assertEqual(self.fetch.call_args_list, [call("KRX")] * attempt)
                self.assertEqual(self.sleep.call_args_list, [call(s) for s in [2, 4, 8, 16][:attempt - 1]])
                payload = json.loads(self.cache.read_text(encoding="utf-8"))
                self.assertEqual(payload["rows"][0]["Code"], "000000")
                self.assertEqual(payload["fetched_at"], NOW.isoformat())

    def test_five_http_failures_use_fresh_cache_without_refreshing_it(self):
        self.write_cache()
        before = self.cache.read_bytes()
        self.fail_fetch()
        pd.testing.assert_frame_equal(universe._get_listing(), self.df)
        self.assertEqual(self.fetch.call_count, 5)
        self.assertEqual(self.sleep.call_args_list, [call(2), call(4), call(8), call(16)])
        self.assertEqual(self.cache.read_bytes(), before)
        self.assertIn("Using KRX universe cache", self.log.call_args.args[0])

    def test_exactly_fourteen_days_is_allowed(self):
        self.write_cache(age=dt.timedelta(days=14))
        self.fail_fetch()
        self.assertEqual(len(universe._get_listing()), 1200)

    def test_stale_or_future_cache_fails(self):
        self.fail_fetch()
        for age in (dt.timedelta(days=14, microseconds=1), dt.timedelta(days=-1)):
            with self.subTest(age=age):
                self.write_cache(age=age)
                with self.assertRaisesRegex(RuntimeError, "outside the allowed 0-14 days"):
                    universe._get_listing()

    def test_missing_cache_has_actionable_error(self):
        self.fail_fetch()
        with self.assertRaisesRegex(RuntimeError, "failed after 5 attempts.*HTTP Error 404.*within 14 days"):
            universe._get_listing()

    def test_corrupt_cache_fails_clearly(self):
        self.fail_fetch()
        valid = self.write_cache()
        for payload in ("{", "[]", json.dumps({**valid, "version": 2}),
                        json.dumps({**valid, "fetched_at": "invalid"}),
                        json.dumps({**valid, "fetched_at": None}),
                        json.dumps({**valid, "fetched_at": "2026-09-10T00:00:00"}),
                        json.dumps({**valid, "rows": {}}),
                        json.dumps({**valid, "rows": ["bad row"]})):
            with self.subTest(payload=payload[:60]):
                self.cache.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "No valid universe cache"):
                    universe._get_listing()

    def test_missing_columns_retry_without_overwriting_good_cache(self):
        self.write_cache()
        before = self.cache.read_bytes()
        for column in universe.REQUIRED_COLUMNS:
            with self.subTest(column=column):
                self.fetch.reset_mock()
                self.fetch.return_value = self.df.drop(columns=column)
                pd.testing.assert_frame_equal(universe._get_listing(), self.df)
                self.assertEqual(self.fetch.call_count, 5)
                self.assertEqual(self.cache.read_bytes(), before)

    def test_incomplete_responses_are_rejected(self):
        one_market = self.df.copy()
        one_market["Market"] = "KOSPI"
        no_prices = self.df.copy()
        no_prices["Close"] = None
        too_few_kospi = self.df.copy()
        too_few_kospi.loc[199:, "Market"] = "KOSDAQ"
        for df in (None, pd.DataFrame(), self.df.iloc[:999], one_market, no_prices, too_few_kospi):
            with self.subTest(kind=type(df), size=None if df is None else len(df)):
                self.fetch.return_value = df
                with self.assertRaisesRegex(RuntimeError, "No valid universe cache"):
                    universe._get_listing()
                self.assertFalse(self.cache.exists())

    def test_invalid_row_values_are_rejected(self):
        for col, value in (("Code", ""), ("Code", 1234), ("Code", "000010"),
                           ("Name", None), ("Market", " "),
                           ("Close", "broken"), ("Marcap", float("inf"))):
            with self.subTest(column=col, value=value):
                invalid = self.df.astype(object)
                invalid.loc[0, col] = value
                with self.assertRaises((ValueError, TypeError)):
                    universe._validate_listing(invalid)

    def test_invalid_cached_listing_is_also_rejected(self):
        self.fail_fetch()
        for df in (self.df.iloc[:999], self.df.drop(columns="Close")):
            self.write_cache(df=df)
            with self.assertRaisesRegex(RuntimeError, "No valid universe cache"):
                universe._get_listing()

    def test_invalid_live_response_can_recover_on_retry(self):
        self.fetch.side_effect = [self.df.iloc[:5], self.df]
        self.assertEqual(len(universe._get_listing()), 1200)
        self.sleep.assert_called_once_with(2)

    def test_cache_write_failure_keeps_fresh_result_and_old_cache(self):
        self.write_cache()
        before = self.cache.read_bytes()
        with patch.object(universe.os, "replace", side_effect=PermissionError("read only")):
            pd.testing.assert_frame_equal(universe._get_listing(), self.df)
        self.fetch.assert_called_once_with("KRX")
        self.sleep.assert_not_called()
        self.assertEqual(self.cache.read_bytes(), before)
        self.assertEqual(list(self.cache.parent.glob("*.tmp")), [])
        self.assertIn("Could not save", self.log.call_args.args[0])

    def test_cache_is_independent_of_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.cache.parent.parent)
            universe._get_listing()
        finally:
            os.chdir(previous)
        self.assertTrue(self.cache.is_file())

    def test_build_preserves_selection_filters_order_labels_and_types(self):
        self.df.loc[0, "Name"] = "제외 스팩"
        self.df.loc[1, "Name"] = "제외 리츠"
        self.df.loc[2, "Name"] = "제외우"
        self.df.loc[3, "Code"] = "999991"
        self.df.loc[4, "Close"] = 999
        self.df.loc[5, "Marcap"] = 0
        self.df.loc[6, "Close"] = None
        self.df.loc[7, "Marcap"] = None
        self.df.loc[8, "Name"] = "제외우B"
        self.df.loc[9, "Name"] = "제외우C"
        self.df.loc[10, "Market"] = "KONEX"
        self.df.loc[11, "Close"] = 1000
        self.df.loc[500, "Market"] = "KOSDAQ GLOBAL"
        selected = universe.build(kospi_n=2, kosdaq_n=1)
        self.assertEqual(selected, [
            ("000110", "Stock 11", "코스피200", 1189000000),
            ("000120", "Stock 12", "코스피200", 1188000000),
            ("005000", "Stock 500", "코스닥150", 700000000),
        ])
        self.assertEqual(universe.build(1, 0, min_price=5000), [selected[1]])
        self.assertEqual(universe.build(0, 0), [])
        default = universe.build()
        self.assertEqual(sum(row[2] == "코스피200" for row in default), 200)
        self.assertEqual(sum(row[2] == "코스닥150" for row in default), 150)
        self.fail_fetch()
        self.assertEqual(universe.build(kospi_n=2, kosdaq_n=1), selected)


if __name__ == "__main__":
    unittest.main()
