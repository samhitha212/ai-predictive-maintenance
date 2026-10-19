"""Batch telemetry ingestion using the model's existing four-feature schema."""

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from model import FEATURES, validate_telemetry


class IngestionError(ValueError):
    """An input source contains malformed or invalid telemetry."""


def ingest_records(records):
    """Validate an iterable of records and return ordered float feature columns.

    Records must contain exactly FEATURES and numeric values (not strings).
    Empty batches are rejected. Validation is all-or-nothing; the source is not
    modified, and the first invalid record raises a one-based row error.
    """
    if isinstance(records, (Mapping, str, bytes)):
        raise IngestionError("Expected an iterable of telemetry records.")
    try:
        iterator = iter(records)
    except TypeError as error:
        raise IngestionError("Expected an iterable of telemetry records.") from error

    cleaned = []
    for row_number, record in enumerate(iterator, start=1):
        if not isinstance(record, Mapping):
            raise IngestionError(f"Row {row_number}: expected a telemetry object.")
        try:
            cleaned.append(validate_telemetry(dict(record)))
        except (ValueError, OverflowError) as error:
            raise IngestionError(f"Row {row_number}: {error}") from error
    if not cleaned:
        raise IngestionError("Telemetry input must contain at least one record.")
    return pd.DataFrame(cleaned, columns=FEATURES, dtype=float)


def load_csv(path):
    """Load a UTF-8 CSV path with exactly the four feature headers.

    Header order may vary. Numeric CSV cells are converted to floats before
    shared validation. Duplicate headers and irregular rows are rejected.
    File-system errors propagate to the caller.
    """
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            headers = reader.fieldnames
            if (
                headers is None
                or len(headers) != len(FEATURES)
                or set(headers) != set(FEATURES)
            ):
                raise IngestionError(f"CSV headers must contain exactly: {FEATURES}")

            def numeric_records():
                for row_number, row in enumerate(reader, start=1):
                    if None in row or any(value is None for value in row.values()):
                        raise IngestionError(
                            f"Row {row_number}: CSV field count does not match headers."
                        )
                    converted = {}
                    for name in FEATURES:
                        try:
                            converted[name] = float(row[name])
                        except (ValueError, OverflowError) as error:
                            raise IngestionError(
                                f"Row {row_number}: {name} must be numeric."
                            ) from error
                    yield converted

            return ingest_records(numeric_records())
    except (csv.Error, UnicodeError) as error:
        raise IngestionError(f"Invalid UTF-8 CSV input: {error}") from error


def _unique_json_object(pairs):
    record = {}
    for name, value in pairs:
        if name in record:
            raise IngestionError(f"Duplicate JSON field: {name}")
        record[name] = value
    return record


def load_json(path):
    """Load a UTF-8 JSON array of numeric telemetry objects from a local path.

    Duplicate object keys are rejected. File-system errors propagate unchanged.
    """
    try:
        with Path(path).open(encoding="utf-8-sig") as source:
            records = json.load(source, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise IngestionError(f"Invalid UTF-8 JSON input: {error}") from error
    if not isinstance(records, list):
        raise IngestionError("JSON input must be an array of telemetry objects.")
    return ingest_records(records)
