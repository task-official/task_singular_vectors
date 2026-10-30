"""Small local run bundles, durable validation journals and exclusive results.

Only files owned by the newly created run directory are written. Historical
results, input weights and datasets are never modified. A resume requires the
same resolved config, source code, checkpoint and bound validation identities.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import uuid
import warnings

import torch
from safetensors.torch import load_file, save_file

from src.core.checkpoints import tensor_fingerprint, file_sha256
from src.core.types import MergeOutput


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        raise TypeError("Tensor arrays belong in safetensors trace files, not result JSON")
    if hasattr(value, "item"):
        return value.item()
    return value


def identity_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_provenance(root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    files = sorted({*root.glob("*.py"), *root.glob("src/**/*.py"),
                    *root.glob("config/**/*.yaml")})
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    # Method/config additions should not invalidate a comparison when the
    # shared input, numerical and evaluation infrastructure is unchanged.
    common_files = {root / "src/__init__.py", root / "src/selection/search.py"}
    for package in ("core", "evaluation", "datasets", "models"):
        common_files.update((root / "src" / package).glob("**/*.py"))
    common_files.update(root / "src/experiments" / name for name in
                        ("config.py", "inputs.py", "runner.py", "storage.py"))
    common = hashlib.sha256()
    for path in sorted(common_files):
        if path.is_file():
            common.update(path.relative_to(root).as_posix().encode())
            common.update(path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True,
                                            stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    import scipy
    import numpy
    import safetensors
    return {"commit": commit, "dirty": dirty, "source_sha256": digest.hexdigest(),
            "common_sha256": common.hexdigest(),
            "mkl_threading_layer": os.environ.get("MKL_THREADING_LAYER"),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "gpu_devices": [{"name": torch.cuda.get_device_name(i),
                             "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                             "capability": list(torch.cuda.get_device_capability(i))}
                            for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "cuda_runtime": torch.version.cuda, "numpy": numpy.__version__,
            "scipy": scipy.__version__, "safetensors": safetensors.__version__}


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(jsonable(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_weights(path: Path, weights: Mapping[str, torch.Tensor]) -> str:
    fingerprint = tensor_fingerprint(weights)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        save_file({key: value.detach().cpu().contiguous() for key, value in weights.items()}, str(temporary))
        if tensor_fingerprint(load_file(str(temporary), device="cpu")) != fingerprint:
            raise ValueError("Saved model does not match the validated weights")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return fingerprint


def split_trace(trace: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Keep exact observed arrays outside JSON without inventing missing trace."""
    tensors = {}

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            key = "/".join(path)
            tensors[key] = value.detach().cpu().contiguous()
            return {"tensor": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Trace mapping keys must be strings")
            # Escape individual path components (JSON Pointer convention), so
            # a key "layer/a" cannot collide with nested keys "layer", "a".
            return {key: walk(item, (*path, key.replace("~", "~0").replace("/", "~1")))
                    for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [walk(v, (*path, str(i))) for i, v in enumerate(value)]
        return jsonable(value)

    return walk(trace, ("trace",)), tensors


class RunStorage:
    """An exclusive run directory with resumable, validation-only state."""

    def __init__(self, output_root: str | Path, identity: Mapping[str, Any], *,
                 run_dir: str | Path | None = None, resume: bool = False):
        self.identity = jsonable(identity)
        self.identity_sha256 = identity_hash(self.identity)
        if run_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = Path(output_root) / f"{stamp}_{self.identity_sha256[:10]}_{uuid.uuid4().hex[:6]}"
        self.path = Path(run_dir)
        self.journal = self.path / "trials.jsonl"
        if resume:
            if not self.path.is_dir():
                raise FileNotFoundError(self.path)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.mkdir(exist_ok=False)
        self._lock = (self.path / ".lock").open("a+")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError("Another process owns this run directory") from None
        try:
            # Check only after taking ownership: another process may have
            # completed between the caller's resume request and lock acquisition.
            if (self.path / "result.json").exists():
                raise FileExistsError("A completed result cannot be resumed or overwritten")
            if resume:
                self.progress = json.loads((self.path / "progress.json").read_text())
                if self.progress.get("identity_sha256") != self.identity_sha256:
                    raise ValueError("Resume input/config/source/validation identity does not match")
                payload = {key: value for key, value in self.progress.items() if key != "record_sha256"}
                if self.progress.get("record_sha256") != identity_hash(payload):
                    raise ValueError("Progress record checksum is inconsistent")
            else:
                self.progress = {"schema_version": 1, "status": "running", "phase": "selection",
                                 "identity_sha256": self.identity_sha256, "identity": self.identity}
                self._write_progress()
            self.history = self._read_history()
        except BaseException:
            self.close()
            raise

    def _write_progress(self):
        payload = {key: value for key, value in self.progress.items() if key != "record_sha256"}
        self.progress["record_sha256"] = identity_hash(payload)
        _atomic_json(self.path / "progress.json", self.progress)

    def _read_history(self) -> list[dict]:
        if not self.journal.exists():
            if self.progress.get("journal_events", 0):
                raise ValueError("Validation journal is missing despite acknowledged events")
            return []
        events = []
        for index, line in enumerate(self.journal.read_text().splitlines()):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Validation journal is incomplete at line {index + 1}; preserve it and use a new run") from error
            if event.get("sequence") != len(events):
                raise ValueError("Validation journal sequence is inconsistent")
            recorded_hash = event.get("event_sha256")
            payload = {key: value for key, value in event.items() if key != "event_sha256"}
            previous_hash = events[-1]["event_sha256"] if events else None
            if event.get("previous_event_sha256") != previous_hash or identity_hash(payload) != recorded_hash:
                raise ValueError("Validation journal checksum is inconsistent")
            events.append(event)
        recorded_count = int(self.progress.get("journal_events", 0))
        if recorded_count > len(events):
            raise ValueError("Validation journal lost acknowledged events")
        if recorded_count and events[recorded_count - 1]["event_sha256"] != self.progress.get("journal_sha256"):
            raise ValueError("Validation journal differs from acknowledged progress")
        return events

    def observe(self, event: dict, output: MergeOutput | None = None) -> None:
        event = dict(event)
        if event.get("sequence") != len(self.history):
            raise ValueError("Validation observer sequence mismatch")
        if output is not None:
            filename = f"best-{event['sequence']:06d}-{uuid.uuid4().hex[:8]}.safetensors"
            if (self.path / filename).exists():
                raise FileExistsError(f"Candidate model already exists: {filename}")
            event["weights_fingerprint"] = _save_weights(self.path / filename, output.weights)
            trace, tensors = split_trace(output.trace)
            event["weights_file"] = filename
            event["trace"] = trace
            if tensors:
                trace_file = f"trace-{event['sequence']:06d}-{uuid.uuid4().hex[:8]}.safetensors"
                save_file(tensors, str(self.path / trace_file))
                event["trace_file"] = trace_file
                event["trace_sha256"] = file_sha256(self.path / trace_file)
        event["previous_event_sha256"] = self.history[-1]["event_sha256"] if self.history else None
        event["event_sha256"] = identity_hash(event)
        # One append after the artifact is safely written. An orphan artifact
        # after a crash is never treated as a completed validation observation.
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(event), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.history.append(event)
        self.progress.update(status="running", phase="selection", journal_events=len(self.history),
                             journal_sha256=event["event_sha256"])
        self._write_progress()
        if output is not None:
            # Only obsolete staging files owned by this new run are removed;
            # historical research files are never involved.
            for old in self.history[:-1]:
                if old.get("is_best") and old.get("weights_file"):
                    (self.path / old["weights_file"]).unlink(missing_ok=True)
                    if old.get("trace_file"):
                        (self.path / old["trace_file"]).unlink(missing_ok=True)

    def resume_output(self) -> MergeOutput | None:
        best = [event for event in self.history if event.get("is_best")]
        if not best:
            return None
        event = best[-1]
        path = self.path / event["weights_file"]
        if not path.is_file():
            raise ValueError("The validated best-model checkpoint is missing; do not reconstruct it silently")
        weights = load_file(str(path), device="cpu")
        if tensor_fingerprint(weights) != event["weights_fingerprint"]:
            raise ValueError("The saved validated model has changed")
        trace = event.get("trace", {})
        if event.get("trace_file"):
            trace_path = self.path / event["trace_file"]
            if not trace_path.is_file() or event.get("trace_sha256") != file_sha256(trace_path):
                raise ValueError("The saved trace checksum is inconsistent")
            arrays = load_file(str(trace_path), device="cpu")

            def restore(value):
                if isinstance(value, dict) and "tensor" in value and "shape" in value:
                    tensor = arrays[value["tensor"]]
                    if list(tensor.shape) != value["shape"] or str(tensor.dtype) != value["dtype"]:
                        raise ValueError("The saved trace shape/dtype is inconsistent")
                    return tensor
                if isinstance(value, dict):
                    return {k: restore(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [restore(v) for v in value]
                return value
            trace = restore(trace)
        return MergeOutput(weights, trace)

    def mark(self, phase: str, **values) -> None:
        self.progress.update(phase=phase, **jsonable(values))
        self._write_progress()

    def close(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def finish(self, result: Mapping[str, Any], output: MergeOutput, *, save_weights: bool = True) -> Path:
        destination = self.path / "result.json"
        if destination.exists():
            raise FileExistsError(destination)
        record = dict(result)
        trace, arrays = split_trace(output.trace)
        if arrays:
            save_file(arrays, str(self.path / "trace.safetensors"))
            record["trace_file"] = "trace.safetensors"
            record["trace_sha256"] = file_sha256(self.path / "trace.safetensors")
        record["trace"] = trace
        record["weights_fingerprint"] = tensor_fingerprint(output.weights)
        if save_weights:
            _save_weights(self.path / "selected.safetensors", output.weights)
            record["weights_file"] = "selected.safetensors"
        record["record_sha256"] = identity_hash(record)
        # link() makes publication exclusive even for concurrent writers.
        temporary = self.path / f"result.{uuid.uuid4().hex}.tmp"
        try:
            _atomic_json(temporary, record)
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        # result.json is the commit point. A cleanup failure must not turn an
        # already published valid result into an interrupted experiment.
        try:
            self.mark("complete", status="complete")
            for event in self.history:
                if event.get("weights_file"):
                    (self.path / event["weights_file"]).unlink(missing_ok=True)
                if event.get("trace_file"):
                    (self.path / event["trace_file"]).unlink(missing_ok=True)
        except OSError as error:
            warnings.warn(f"Result published; progress/staging cleanup failed: {error}", RuntimeWarning)
        return destination
