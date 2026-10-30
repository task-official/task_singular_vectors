"""Content-identified CPU tensor sources. Conversion is an explicit prepare step."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import uuid
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .types import MergeInputs


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_fingerprint(weights: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(weights):
        tensor = weights[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class MemoryTensorSource:
    """Useful for small in-memory problems and characterization tests."""

    def __init__(self, weights: Mapping[str, torch.Tensor]):
        self._weights = {name: value.detach().cpu().clone() for name, value in weights.items()}
        self.identity = tensor_fingerprint(self._weights)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._weights))

    def tensor(self, name: str) -> torch.Tensor:
        return self._weights[name]


class SafeTensorSource:
    """Memory-mapped file; no full-model tensor stack is retained."""

    def __init__(self, path: str | Path, expected_sha256: str | None = None):
        self.path = Path(path)
        self.identity = file_sha256(self.path)
        if expected_sha256 is not None and self.identity != expected_sha256:
            raise ValueError(f"Checkpoint digest mismatch: {self.path}")
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            self._keys = tuple(sorted(handle.keys()))
        if not self._keys:
            raise ValueError(f"Empty checkpoint: {self.path}")

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def make_inputs(base, finetuned, task_order=None) -> MergeInputs:
    """Validate sources with an explicit stable parameter order and task order."""
    base = MemoryTensorSource(base) if isinstance(base, Mapping) else base
    finetuned = {
        task: MemoryTensorSource(source) if isinstance(source, Mapping) else source
        for task, source in finetuned.items()
    }
    order = tuple(finetuned) if task_order is None else tuple(task_order)
    if not order or len(set(order)) != len(order) or set(order) != set(finetuned):
        raise ValueError("task_order must name each fine-tuned source exactly once")
    names = base.keys()
    for task in order:
        source = finetuned[task]
        if source.keys() != names:
            raise ValueError(f"Checkpoint parameter keys differ for {task}")
        for name in names:
            original, tuned = base.tensor(name), source.tensor(name)
            if original.shape != tuned.shape or original.dtype != tuned.dtype:
                raise ValueError(f"Checkpoint shape/dtype mismatch: {task}/{name}")
            if original.is_floating_point():
                if not torch.isfinite(original).all() or not torch.isfinite(tuned).all():
                    raise ValueError(f"Non-finite checkpoint tensor: {task}/{name}")
            elif not torch.equal(original, tuned):
                raise ValueError(f"Non-floating buffer differs: {task}/{name}")
    identity = hashlib.sha256(json.dumps({
        "base": base.identity,
        "tasks": [(task, finetuned[task].identity) for task in order],
        "parameters": names,
    }, sort_keys=True).encode()).hexdigest()
    return MergeInputs(base, finetuned, order, names, identity)


def prepared_path(source: str | Path, cache_dir: str | Path) -> Path:
    source = Path(source).expanduser()
    if source.suffix == ".safetensors":
        return source
    digest = file_sha256(source)
    return Path(cache_dir) / "checkpoints" / f"{digest}.safetensors"


def open_prepared(source: str | Path, cache_dir: str | Path) -> SafeTensorSource:
    source = Path(source).expanduser()
    destination = prepared_path(source, cache_dir)
    if source.suffix == ".safetensors":
        return SafeTensorSource(source)
    metadata_path = destination.with_suffix(".json")
    if not destination.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Not prepared: {source}. Run prepare.py first.")
    metadata = json.loads(metadata_path.read_text())
    if metadata["source_sha256"] != file_sha256(source):
        raise ValueError(f"Prepared source changed: {source}")
    return SafeTensorSource(destination, metadata["converted_sha256"])


def prepare_checkpoint(source: str | Path, cache_dir: str | Path) -> Path:
    """Serialize preparation and recover source-verified incomplete conversions."""
    source = Path(source).expanduser()
    destination = prepared_path(source, cache_dir)
    if source.suffix == ".safetensors":
        SafeTensorSource(source)
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _prepare_checkpoint(source, cache_dir)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _prepare_checkpoint(source: str | Path, cache_dir: str | Path) -> Path:
    """Convert a trusted local PyTorch checkpoint without changing any tensor.

    PyTorch pickle files can execute code; only use trusted author/local artifacts.
    The original file is never overwritten. No network calls are made.
    """
    source = Path(source).expanduser()
    destination = prepared_path(source, cache_dir)
    if source.suffix == ".safetensors":
        SafeTensorSource(source)
        return source
    if destination.exists() and destination.with_suffix(".json").exists():
        open_prepared(source, cache_dir)
        return destination
    digest = file_sha256(source)
    metadata_path = destination.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("source_sha256") != digest:
            raise ValueError("Orphan preparation metadata belongs to another source")
    checkpoint = torch.load(source, map_location="cpu")
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a state_dict or a module checkpoint")
    checkpoint = {key: value for key, value in checkpoint.items() if key != "model_name"}
    if not checkpoint or not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise TypeError("Checkpoint contains non-tensor state entries")
    weights = {key: value.detach().cpu().contiguous().clone() for key, value in checkpoint.items()}
    del checkpoint
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        save_file(weights, str(temporary))
        with safe_open(str(temporary), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(weights):
                raise ValueError("Conversion changed parameter names")
            for name, original in weights.items():
                converted = handle.get_tensor(name)
                if original.dtype != converted.dtype or not torch.equal(original, converted):
                    raise ValueError(f"Conversion changed tensor {name}")
        if file_sha256(source) != digest:
            raise ValueError("Source checkpoint changed during conversion")
        # link() is exclusive, unlike replace(); simultaneous preparation cannot overwrite.
        if destination.exists():
            # An interruption may have published tensors but not metadata.
            # Recover only after checking every value against the trusted source.
            with safe_open(str(destination), framework="pt", device="cpu") as reader:
                if set(reader.keys()) != set(weights):
                    raise ValueError("Incomplete checkpoint keys differ from source")
                for name, value in weights.items():
                    restored = reader.get_tensor(name)
                    if restored.dtype != value.dtype or not torch.equal(restored, value):
                        raise ValueError(f"Incomplete checkpoint differs from source: {name}")
        else:
            os.link(temporary, destination)
        metadata = {
            "format_version": 1,
            "source_sha256": digest,
            "converted_sha256": file_sha256(destination),
            "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in weights.items()},
        }
        temporary_metadata = temporary.with_suffix(".json")
        try:
            with temporary_metadata.open("x") as handle:
                json.dump(metadata, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_metadata, metadata_path)
        finally:
            temporary_metadata.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
