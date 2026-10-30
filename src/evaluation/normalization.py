"""Per-checkpoint, split, head and preprocessing normalization cache.

Historical results/single_task JSON files are never rewritten or guessed to be
compatible with the current experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from src.core.types import PreparedExperiment
from src.datasets.adapter import file_digest

NORMALIZER_VERSION = "individual-ft-v2"


def _identity_key(identity: dict) -> str:
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path, identity: dict) -> float:
    data = json.loads(path.read_text())
    # Compare canonical encodings because tuples become lists in JSON.
    if _identity_key(data["identity"]) != _identity_key(identity):
        raise ValueError(f"Normalization identity mismatch: {path}")
    payload = {key: value for key, value in data.items() if key != "record_sha256"}
    if data.get("record_sha256") != _identity_key(payload):
        raise ValueError(f"Normalization record checksum mismatch: {path}")
    accuracy = float(data["accuracy"])
    if not math.isfinite(accuracy) or not 0 < accuracy <= 1:
        raise ValueError(f"Invalid normalization accuracy in {path}")
    return accuracy


def get_baselines(prepared: PreparedExperiment, split: str, evaluator, cache_dir: str | None = None) -> dict[str, float]:
    if split != evaluator.split:
        raise ValueError("Normalizer/evaluator split mismatch")
    directory = Path(cache_dir or prepared.spec.runtime.cache_dir) / "normalization"
    directory.mkdir(parents=True, exist_ok=True)
    absolute_evaluator = evaluator.with_baselines(None)
    results = {}
    for task_id in evaluator.task_ids:
        source = prepared.merge_inputs.finetuned[task_id]
        identity = {
            "version": NORMALIZER_VERSION, "task": task_id, "split": split,
            "implementation_sha256": file_digest(Path(__file__)),
            "finetuned": source.identity,
            "evaluator": {key: value for key, value in evaluator.identity.items() if key not in ("tasks", "baselines")},
            "data": evaluator.identity["tasks"][task_id],
        }
        path = directory / f"{_identity_key(identity)}.json"
        if path.exists():
            results[task_id] = _read(path, identity)
            continue
        weights = {name: source.tensor(name) for name in source.keys()}
        metrics = absolute_evaluator.evaluate(weights, (task_id,)).per_task[task_id]
        accuracy = metrics.top1
        del weights
        if not math.isfinite(accuracy) or not 0 < accuracy <= 1:
            raise ValueError(f"Cannot normalize {task_id}/{split}: individual accuracy is {accuracy}")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".normalization-", suffix=".json", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w") as handle:
                record = {"identity": identity, "accuracy": accuracy,
                          "correct": metrics.correct, "total": metrics.total}
                record["record_sha256"] = _identity_key(record)
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        results[task_id] = _read(path, identity)
    return results
