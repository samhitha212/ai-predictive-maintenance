# Sentinel: AI-Driven Predictive Maintenance with Blockchain Logging

An executable portfolio prototype combining machine-learning inference, an interactive telemetry dashboard, and a cryptographic prediction audit trail.

**Stack:** Python · Scikit-Learn · Streamlit · Pandas · NumPy · SHA-256

## Author and Maintainer

**[samhitha212](https://github.com/samhitha212)** is the project's author and sole maintainer.

## Executive Project Summary

Hardware failures can interrupt services, damage availability, and increase maintenance costs. Sentinel demonstrates how CPU utilization, temperature, disk health, and uptime can feed a failure-risk classifier, with each operator-selected prediction recorded in a verifiable ledger.

A conventional database alone does not prove that privileged users have not edited or deleted historical logs. Access controls, append-only storage, signed events, and database auditing can mitigate this; databases are not inherently incapable of secure audit logging.

A hash-linked blockchain adds explicit tamper evidence: modifying a recorded snapshot changes its hash and invalidates subsequent links. Proof-of-work requires additional computation to rebuild modified history. Distributed consensus and externally anchored checkpoints can make unilateral rewriting harder.

This prototype implements the hash-linked, proof-of-work mechanism locally. It is **not** a distributed consensus network, permanent storage service, or guarantee of absolute immutability.

## Portfolio Context

This standalone educational reconstruction illustrates the hardware-failure prediction and blockchain-logging concept described in my CR Rao AIMSCS ML internship experience. It uses newly generated synthetic data, not employer code, confidential telemetry, or independently verified internship performance results.

## System Architecture

    +-------------------------------------------+
    | Sensor Input: interactive sidebar sliders |
    | CPU | Temperature | Disk health | Uptime   |
    +---------------------+---------------------+
                          |
                          v
    +-------------------------------------------+
    | Schema, finite-value, and range validation|
    +---------------------+---------------------+
                          |
                          v
    +-------------------------------------------+
    | AI Inference Model: RandomForestClassifier|
    | Failure-class probability -> risk percent |
    +---------------------+---------------------+
                          |
                          v
    +-------------------------------------------+
    | Operator reviews risk and commits alert   |
    +---------------------+---------------------+
                          |
                          v
    +-------------------------------------------+
    | Cryptographic Block Serialization        |
    | Canonical JSON + UTC time + previous hash|
    +---------------------+---------------------+
                          |
                          v
    +-------------------------------------------+
    | SHA-256 proof-of-work: three leading zeros|
    +---------------------+---------------------+
                          |
                          v
    +-------------------------------------------+
    | Session ledger -> DataFrame / JSON export |
    | Audit: genesis, hashes, links, and work    |
    +-------------------------------------------+

## Rapid Local Setup

Use **CPython 3.11 or 3.12**. The exact direct-dependency pins target this range; no dependency set is universally compatible with every Python release or operating system.

1. Clone the repository:

       git clone https://github.com/samhitha212/ai-predictive-maintenance.git
       cd ai-predictive-maintenance

   If you already have the project locally, open its directory and skip cloning.

2. Create and activate an isolated environment on macOS/Linux:

       python3 -m venv .venv
       source .venv/bin/activate

   On Windows PowerShell:

       py -3.12 -m venv .venv
       .venv\Scripts\Activate.ps1

3. Install dependencies:

       python -m pip install --upgrade pip
       python -m pip install -r requirements.txt
       python -m pip check

4. Launch the dashboard:

       python -m streamlit run app.py

5. Open `http://localhost:8501`. Training runs automatically on first launch and is cached for subsequent widget reruns.

No wallet, API key, cloud account, external dataset, model download, or blockchain node is required.

## Demonstration Walkthrough

1. Keep the default telemetry to inspect a routine operating state.
2. Set CPU to 96%, temperature to 94°C, disk health to 15, and uptime to 12,000 hours to simulate a stressed machine.
3. Observe the model's estimated risk and operational guidance.
4. Click **Commit Alert to Immutable Blockchain Ledger** to mine and append a prediction block. The button retains the requested product label; the actual guarantee is local tamper evidence.
5. Open **Blockchain Ledger Explorer**, inspect the snapshot and hashes, and run the integrity audit.
6. Download the ledger JSON before ending the session if you want a record.

## Local Telemetry Ingestion

`ingestion.py` provides `load_csv(path)`, `load_json(path)`, and
`ingest_records(records)` for batch inference inputs. Each returns a pandas
DataFrame with float columns in the model's feature order:

- `cpu_utilization`: 0–100 percent.
- `core_temperature`: 30–100 degrees Celsius.
- `disk_health_score`: 0–100.
- `uptime_hours`: 0–20,000 hours.

CSV files must have exactly these four headers in any order, followed by numeric
cells. JSON files must contain an array of objects with exactly these fields and
numeric values; in-memory ingestion accepts an iterable of equivalent mappings.
UTF-8 files with or without a BOM are supported.

Empty batches, missing or unexpected fields, duplicate CSV headers or JSON keys,
booleans, missing values, non-finite numbers, and out-of-range values are rejected.
Validation is all-or-nothing: invalid records raise `IngestionError`, with a
one-based data-row number for record validation errors. CSV row numbers exclude
the header and blank lines. File access errors such as `FileNotFoundError` are
left unchanged. Numeric strings are converted only by the CSV loader.

For example, save a local `telemetry.csv` containing:

    cpu_utilization,core_temperature,disk_health_score,uptime_hours
    48,58,92,2400
    96,94,15,12000

Then use it from Python:

    from ingestion import load_csv
    from model import PredictiveMaintenanceModel

    data = load_csv("telemetry.csv")
    model = PredictiveMaintenanceModel()
    model.train()
    risks = [model.predict_proba(row) for row in data.to_dict(orient="records")]

This initial module loads local batches into memory. It does not ingest labels,
machine identifiers, or timestamps, retrain from imported data, persist records,
or connect to live sensors. The dashboard continues to use its sidebar sliders;
file-upload integration is not yet implemented.

## Machine Learning Pipeline

`generate_synthetic_telemetry(n_samples=1200)` uses a fixed seed for reproducibility. Temperature is correlated with CPU utilization, and disk health trends downward with uptime. Eighteen percent of rows receive deliberate stress spikes. Labels identify combined CPU/thermal/disk stress, severe disk deterioration with long uptime, or extreme thermal load. Independent label flips affect approximately 1.5% of rows.

The Random Forest uses 250 trees, a minimum leaf size of two, and `balanced_subsample` class weighting. A stratified 75%/25% train/test split preserves minority failure representation without training on the test partition.

Evaluation reports positive-class precision, recall, and F1 at the classifier's default decision rule. The dashboard's operational risk bands are separate demonstration thresholds: low below 30%, medium from 30% to below 70%, and high from 70%.

`predict_proba()` returns the unrounded model probability multiplied by 100. This is an estimated score, not an exact or calibrated real-world probability. No forecast horizon or remaining-useful-life estimate is established by these synthetic labels.

## Project Motivation and Engineering Decisions

### Addressing class imbalance

Failures are usually less frequent than normal operating conditions. A classifier can achieve attractive accuracy while missing the events that matter. Controlled minority stress examples ensure both classes are available; stratification preserves their representation; class weighting penalizes minority errors; precision, recall, and F1 make failure-class behavior visible.

These measures mitigate imbalance in the demonstration, but do not prove robustness against realistic rare-event rates, novel faults, or distribution shifts. A production evaluation should use real, timestamped, machine-grouped data and avoid machine or temporal leakage.

### Securing audit logs

Canonical JSON fixes key order and whitespace before SHA-256 hashing. Each prediction block binds its index, UTC timestamp, machine identifier, telemetry, risk, previous hash, and numeric nonce. Non-genesis hashes must start with three hexadecimal zeros; the fixed genesis block is explicitly exempt.

The audit recomputes hashes, checks sequential indices, validates prediction fields, verifies proof-of-work, checks previous-hash continuity, and compares genesis against its fixed definition. Appending is refused if the existing ledger fails validation.

The mechanism detects ordinary edits, middle-block deletion, reordering, and inconsistent links. It cannot detect removal of the newest valid suffix without an external checkpoint. An attacker controlling the process can also rewrite and re-mine the entire chain. Sensor truth, identity, timestamps, authorization, and model provenance are not cryptographically attested.

### Operational simplicity

Streamlit keeps the interface compact, while session state preserves each browser session's ledger across normal widget reruns. A cached model avoids repeated training. The ledger is not shared between browser sessions and is lost when the session or process ends; JSON export is a download, not automatic persistence or restoration.

## Verification

Local validation on CPython 3.12.4 with the pinned direct dependencies produced F1 **0.9541**, precision **0.9811**, and recall **0.9286** on 300 held-out synthetic rows. These are simulator results, not real hardware benchmarks.

Streamlit AppTest checks passed for startup, both tabs, committing blocks, ledger retention across reruns, high-risk slider inputs, successful audits, tamper alerts, and refusal to append to a corrupted ledger. The local test environment emitted a non-blocking Streamlit cache coroutine warning.

Run standalone training and syntax checks:

    python model.py
    python -m py_compile model.py blockchain.py app.py ingestion.py

Run the ingestion regression suite (including compatibility with model inference):

    python -m unittest discover -s tests -v

For an interactive tamper demonstration, start `python` in this directory and execute:

    from blockchain import Blockchain
    ledger = Blockchain()
    snapshot = dict(cpu_utilization=96, core_temperature=94, disk_health_score=15, uptime_hours=12000)
    ledger.add_prediction_block("SERVER-001", snapshot, 91.5)
    assert ledger.is_chain_valid()
    ledger.ledger[1].telemetry_snapshot["disk_health_score"] = 99.0
    assert not ledger.is_chain_valid()

The manually supplied risk in this integrity-only example is not a model evaluation result.

## Repository Guide

- `model.py`: synthesis, input validation, training, probability inference, and evaluation.
- `ingestion.py`: validated local CSV, JSON, and in-memory telemetry ingestion.
- `tests/test_ingestion.py`: ingestion validation and model-compatibility tests.
- `blockchain.py`: block serialization, proof-of-work, append validation, and chain auditing.
- `app.py`: two-tab dashboard, interactive telemetry, ledger explorer, and JSON download.
- `requirements.txt`: exact direct-dependency versions.
- `.gitignore`: Python artifacts, environments, model files, secrets, and OS caches.
- `README.md`: architecture, setup, demonstration, methodology, and limitations.

## Production Evolution

Future work, not implemented in this prototype: authenticated telemetry ingestion, durable append-only storage, signed machine events, externally anchored ledger checkpoints, multi-node consensus, recorded model versions, calibrated probabilities, temporal evaluation on real failures, alert deduplication, monitoring, and access-controlled deployment.

This prototype demonstrates the complete local integration path without claiming production readiness or real-world failure-prediction performance.
