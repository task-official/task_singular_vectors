"""Compare saved records by identities, never by guessing filenames."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.experiments.storage import identity_hash


def load_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    results = []
    for source in paths:
        path = Path(source)
        if path.is_dir():
            path = path / "result.json"
        record = json.loads(path.read_text())
        if record.get("schema_version") != 1 or record.get("status") != "complete" or "test" not in record:
            raise ValueError(f"Not a complete new-format experiment: {path}; import legacy results explicitly")
        if "record_sha256" in record:
            payload = {key: value for key, value in record.items() if key != "record_sha256"}
            if identity_hash(payload) != record["record_sha256"]:
                raise ValueError(f"Result record checksum mismatch: {path}")
        record["result_path"] = str(path)
        results.append(record)
    return results


def comparison_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    config = result["config"]
    runtime = config.get("runtime", {})
    selection = config.get("selection", {})
    mode = selection.get("mode", "fixed")
    return {
        "source_type": result.get("source_type", "local"),
        "common_revision": result.get("provenance", {}).get("common_sha256",
                             result.get("provenance", {}).get("source_sha256")),
        "numerical_environment": {key: result.get("provenance", {}).get(key) for key in
                                  ("python", "torch", "numpy", "scipy", "safetensors", "cuda_runtime",
                                   "cpu_threads", "cpu_interop_threads", "mkl_threading_layer", "gpu_devices")},
        "protocol": config.get("protocol"),
        "model": config.get("model"),
        "benchmark": config.get("benchmark"),
        "inputs": result["input_identity"],
        "validation": result.get("validation_identity"),
        "test": result.get("test_identity"),
        "runtime": {key: runtime.get(key) for key in ("device", "compute_dtype", "batch_size", "seed")},
        # Grid and BO may propose different, method-appropriate candidates,
        # but must share the same data-access category and image allowance.
        "selection": {"category": "fixed" if mode == "fixed" else "tuned",
                      "max_validation_images": selection.get("max_validation_images"),
                      "seed": selection.get("seed"),
                      "objective": selection.get("objective", "macro_normalized_top1")},
    }


def check_comparability(results: list[Mapping[str, Any]], *, allow_mixed: bool = False) -> dict[str, Any]:
    identities = [comparison_identity(result) for result in results]
    groups = [identity_hash(identity) for identity in identities]
    matching = len(set(groups)) <= 1
    if not matching and not allow_mixed:
        raise ValueError("Results use different checkpoints, splits, heads, preprocessing, protocols or runtime; use allow_mixed to label them separately")
    return {"comparable": matching, "groups": groups, "identities": identities}


def make_comparison_table(results: list[Mapping[str, Any]], *, allow_mixed: bool = False) -> list[dict[str, Any]]:
    comparison = check_comparability(results, allow_mixed=allow_mixed)
    rows = []
    for result, group in zip(results, comparison["groups"]):
        method = result["method"]
        rows.append({
            "model": result["config"].get("model"),
            "benchmark": result["config"].get("benchmark"),
            "method": method["id"] if isinstance(method, Mapping) else method,
            "implementation": method.get("implementation", "unknown") if isinstance(method, Mapping) else "unknown",
            "source_revision": result.get("provenance", {}).get("source_sha256", "unknown"),
            "common_revision": result.get("provenance", {}).get("common_sha256", "not_recorded"),
            "record_integrity": "checksummed" if "record_sha256" in result else "not_recorded",
            "protocol": result["config"].get("protocol"),
            "selection_mode": result["config"].get("selection", {}).get("mode", "fixed"),
            "validation_images": sum(int(e.get("expected_images", 0))
                                     for e in result.get("validation_history", [])
                                     if e.get("event") == "evaluation_start"),
            "comparison_group": group,
            "avg_top1": result["test"]["avg_top1"],
            "avg_normalized_top1": result["test"].get("avg_normalized_top1"),
            "per_task": json.dumps(result["test"].get("per_task", {}), sort_keys=True),
            "selected_params": json.dumps(result["selected_params"], sort_keys=True),
            "weights_fingerprint": result["weights_fingerprint"],
            "source": result.get("result_path", result.get("run_dir", "")),
        })
    return rows


def export_csv(table: list[Mapping[str, Any]], path: str | Path) -> Path:
    if not table:
        raise ValueError("Cannot export an empty comparison")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return path
