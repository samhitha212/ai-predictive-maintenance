"""Run with: python -m unittest discover -s tests -v."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ingestion import IngestionError, ingest_records, load_csv, load_json
from model import BOUNDS, FEATURES, PredictiveMaintenanceModel


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(zip(FEATURES, [48, 58, 92, 2400]))
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_source(self, content, suffix):
        path = self.directory / f"telemetry.{suffix}"
        path.write_text(content, encoding="utf-8")
        return path

    def csv_source(self, values):
        return ",".join(FEATURES) + "\n" + values + "\n"

    def test_records_are_ordered_float_columns_without_mutating_input(self):
        original = self.record.copy()
        source = dict(reversed(list(self.record.items())))
        frame = ingest_records([source])
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(frame.iloc[0].to_dict(), original)
        self.assertTrue(all(dtype == float for dtype in frame.dtypes))
        frame.iloc[0, 0] = 0
        self.assertEqual(source, original)

    def test_generator_preserves_row_order(self):
        records = (dict(self.record, cpu_utilization=value) for value in [10, 90])
        self.assertEqual(ingest_records(records)["cpu_utilization"].tolist(), [10, 90])

    def test_bounds_are_inclusive(self):
        for endpoint in [0, 1]:
            record = {name: bounds[endpoint] for name, bounds in BOUNDS.items()}
            self.assertEqual(len(ingest_records([record])), 1)

    def test_invalid_container_types(self):
        for value in [None, 10, "telemetry", b"telemetry", self.record]:
            with self.subTest(value=value), self.assertRaises(IngestionError):
                ingest_records(value)

    def test_empty_batch(self):
        with self.assertRaisesRegex(IngestionError, "at least one"):
            ingest_records([])

    def test_non_object_row_has_row_number(self):
        with self.assertRaisesRegex(IngestionError, "Row 2"):
            ingest_records([self.record, None])

    def test_missing_and_extra_fields(self):
        missing = self.record.copy()
        del missing[FEATURES[0]]
        for record in [missing, dict(self.record, failure=1)]:
            with self.subTest(record=record), self.assertRaisesRegex(
                IngestionError, "Row 1"
            ):
                ingest_records([record])

    def test_invalid_numeric_values(self):
        for value in [True, None, "48", float("nan"), float("inf"), -1, 101, 10**400]:
            with self.subTest(value=value), self.assertRaisesRegex(
                IngestionError, "Row 2"
            ):
                ingest_records([self.record, dict(self.record, cpu_utilization=value)])

    def test_csv_loads_reordered_headers_and_bom(self):
        source = "\ufeff" + ",".join(reversed(FEATURES)) + "\n2400,92,58,48\n"
        frame = load_csv(self.write_source(source, "csv"))
        pd.testing.assert_frame_equal(frame, ingest_records([self.record]))

    def test_csv_rejects_bad_headers(self):
        for source in [
            "",
            "cpu_utilization\n48\n",
            ",".join(FEATURES + ["failure"]) + "\n48,58,92,2400,0\n",
            ",".join(FEATURES[:-1] + [FEATURES[0]]) + "\n48,58,92,2400\n",
        ]:
            with self.subTest(source=source), self.assertRaisesRegex(
                IngestionError, "headers"
            ):
                load_csv(self.write_source(source, "csv"))

    def test_csv_rejects_header_only(self):
        with self.assertRaisesRegex(IngestionError, "at least one"):
            load_csv(self.write_source(",".join(FEATURES), "csv"))

    def test_csv_rejects_irregular_rows(self):
        for values in ["48,58,92", "48,58,92,2400,extra"]:
            with self.subTest(values=values), self.assertRaisesRegex(
                IngestionError, "Row 1.*field count"
            ):
                load_csv(self.write_source(self.csv_source(values), "csv"))

    def test_csv_rejects_invalid_cells(self):
        for value in ["", "true", "oops", "nan", "inf", "101"]:
            with self.subTest(value=value), self.assertRaisesRegex(
                IngestionError, "Row 1"
            ):
                load_csv(
                    self.write_source(self.csv_source(f"{value},58,92,2400"), "csv")
                )

    def test_csv_rejects_malformed_quoting(self):
        with self.assertRaisesRegex(IngestionError, "Invalid UTF-8 CSV"):
            load_csv(self.write_source(self.csv_source('"48,58,92,2400'), "csv"))

    def test_json_loads_records_with_bom(self):
        path = self.write_source("\ufeff" + json.dumps([self.record]), "json")
        pd.testing.assert_frame_equal(load_json(path), ingest_records([self.record]))

    def test_json_rejects_invalid_syntax(self):
        with self.assertRaisesRegex(IngestionError, "Invalid UTF-8 JSON"):
            load_json(self.write_source("[{", "json"))

    def test_json_requires_array(self):
        for value in [self.record, None, "telemetry", 1]:
            with self.subTest(value=value), self.assertRaisesRegex(
                IngestionError, "array"
            ):
                load_json(self.write_source(json.dumps(value), "json"))

    def test_json_rejects_empty_array_and_invalid_rows(self):
        for value in [[], [None], [dict(self.record, cpu_utilization="48")]]:
            with self.subTest(value=value), self.assertRaises(IngestionError):
                load_json(self.write_source(json.dumps(value), "json"))

    def test_json_rejects_duplicate_keys(self):
        source = '[{"cpu_utilization":48,"cpu_utilization":90}]'
        with self.assertRaisesRegex(IngestionError, "Duplicate JSON field"):
            load_json(self.write_source(source, "json"))

    def test_json_rejects_nonfinite_values(self):
        for value in ["NaN", "Infinity", "1e400"]:
            source = json.dumps([self.record]).replace(
                '"cpu_utilization": 48', f'"cpu_utilization": {value}'
            )
            with self.subTest(value=value), self.assertRaisesRegex(
                IngestionError, "Row 1"
            ):
                load_json(self.write_source(source, "json"))

    def test_loaders_preserve_file_not_found(self):
        for loader in [load_csv, load_json]:
            with self.subTest(loader=loader), self.assertRaises(FileNotFoundError):
                loader(self.directory / "missing")

    def test_loaders_reject_invalid_utf8(self):
        path = self.directory / "invalid"
        path.write_bytes(b"\xff\xfe")
        for loader in [load_csv, load_json]:
            with self.subTest(loader=loader), self.assertRaises(IngestionError):
                loader(path)

    def test_ingested_records_work_with_existing_model(self):
        model = PredictiveMaintenanceModel()
        model.train()
        record = ingest_records([self.record]).to_dict(orient="records")[0]
        risk = model.predict_proba(record)
        self.assertGreaterEqual(risk, 0)
        self.assertLessEqual(risk, 100)


if __name__ == "__main__":
    unittest.main()
