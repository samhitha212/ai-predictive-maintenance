"""Local proof-of-work ledger: tamper-evident, not a distributed blockchain."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from numbers import Real


@dataclass
class Block:
    index: int
    timestamp: str
    machine_id: str
    telemetry_snapshot: dict
    prediction_risk: float
    previous_hash: str
    nonce: int = 0
    hash: str = field(init=False)

    def __post_init__(self):
        self.telemetry_snapshot = dict(self.telemetry_snapshot)
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        payload = {
            "index": self.index,
            "timestamp": self.timestamp,
            "machine_id": self.machine_id,
            "telemetry_snapshot": self.telemetry_snapshot,
            "prediction_risk": self.prediction_risk,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        serialized = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def mine(self, difficulty):
        prefix = "0" * difficulty
        self.hash = self.calculate_hash()
        while not self.hash.startswith(prefix):
            self.nonce += 1
            self.hash = self.calculate_hash()


class Blockchain:
    DIFFICULTY = 3

    def __init__(self):
        self.ledger = [self._genesis()]

    @staticmethod
    def _genesis():
        """Fixed genesis contents; deliberately exempt from proof-of-work."""
        return Block(
            index=0,
            timestamp="2026-01-01T00:00:00+00:00",
            machine_id="GENESIS",
            telemetry_snapshot={},
            prediction_risk=0.0,
            previous_hash="0" * 64,
            nonce=0,
        )

    @staticmethod
    def _validate_risk(risk):
        if isinstance(risk, bool) or not isinstance(risk, Real):
            raise ValueError("Risk must be numeric.")
        risk = float(risk)
        if not math.isfinite(risk) or not 0 <= risk <= 100:
            raise ValueError("Risk must be a finite percentage between 0 and 100.")
        return risk

    @staticmethod
    def _validate_telemetry(telemetry):
        bounds = {
            "cpu_utilization": (0, 100),
            "core_temperature": (30, 100),
            "disk_health_score": (0, 100),
            "uptime_hours": (0, 20000),
        }
        if not isinstance(telemetry, dict) or set(telemetry) != set(bounds):
            raise ValueError("Telemetry must contain exactly the four sensor features.")
        cleaned = {}
        for name, (minimum, maximum) in bounds.items():
            value = telemetry[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric.")
            value = float(value)
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")
            cleaned[name] = value
        return cleaned

    def add_prediction_block(self, machine_id, telemetry, risk):
        if not self.is_chain_valid():
            raise ValueError("Ledger integrity failed; refusing to append.")
        if not isinstance(machine_id, str) or not machine_id.strip():
            raise ValueError("machine_id must be a nonempty string.")
        if len(machine_id.strip()) > 80:
            raise ValueError("machine_id must not exceed 80 characters.")
        block = Block(
            index=len(self.ledger),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            machine_id=machine_id.strip(),
            telemetry_snapshot=self._validate_telemetry(telemetry),
            prediction_risk=self._validate_risk(risk),
            previous_hash=self.ledger[-1].hash,
        )
        block.mine(self.DIFFICULTY)
        self.ledger.append(block)
        return block

    def is_chain_valid(self):
        """Verify genesis, block data, hashes, linkage, indices, and mined work."""
        try:
            if not isinstance(self.ledger, list) or not self.ledger:
                return False
            if asdict(self.ledger[0]) != asdict(self._genesis()):
                return False
            for index, block in enumerate(self.ledger):
                if type(block.index) is not int or block.index != index:
                    return False
                if type(block.nonce) is not int or block.nonce < 0:
                    return False
                if block.hash != block.calculate_hash():
                    return False
                if index == 0:
                    continue
                if block.previous_hash != self.ledger[index - 1].hash:
                    return False
                if not block.hash.startswith("0" * self.DIFFICULTY):
                    return False
                if not isinstance(block.machine_id, str) or not block.machine_id.strip():
                    return False
                if len(block.machine_id) > 80:
                    return False
                timestamp = datetime.fromisoformat(block.timestamp)
                if timestamp.utcoffset() is None:
                    return False
                self._validate_telemetry(block.telemetry_snapshot)
                self._validate_risk(block.prediction_risk)
            return True
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    def to_records(self):
        """Return detached records suitable for tabular display or JSON export."""
        return [asdict(block) for block in self.ledger]
