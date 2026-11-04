"""Compare saved records by identities, never by guessing filenames."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.experiments.storage import identity_hash


METHOD_LABELS = {
    "zeroshot": "Zero-shot", "weight_average": "Weight Averaging",
    "task_arithmetic": "Task Arithmetic", "ties": "TIES-Merging",
    "consensus_ta": "Consensus TA", "tsvm": "TSV-M", "iso_cts": "Iso-CTS",
    "dc_merge": "DC-Merge", "gse_tsv": "GSE-TSV", "sera": "SERA",
}


def method_id(record: Mapping[str, Any]) -> str:
    method = record["method"]
    return method["id"] if isinstance(method, Mapping) else method


def select_methods(results: list[Mapping[str, Any]], methods: list[str]) -> list[Mapping[str, Any]]:
    """Select explicitly named methods, preserving their requested order."""
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a nonempty list without duplicates")
    missing = set(methods) - {method_id(record) for record in results}
    if missing:
        raise ValueError(f"Requested methods not found: {sorted(missing)}")
    return [record for name in methods for record in results if method_id(record) == name]


def comparison_labels(results) -> list[str]:
    """Readable labels; unfamiliar method IDs need no plotting registration."""
    from collections import Counter
    ids = [method_id(record) for record in results]
    repeated = Counter(ids)
    labels = []
    for record, name in zip(results, ids):
        label = METHOD_LABELS.get(name, name)
        if repeated[name] > 1:
            method = record["method"]
            version = method.get("implementation", "unknown") if isinstance(method, Mapping) else "unknown"
            label += f" [{version}] {json.dumps(record.get('selected_params', {}), sort_keys=True)}"
        labels.append(label)
    counts, seen = Counter(labels), Counter()
    for index, label in enumerate(labels):
        if counts[label] > 1:
            seen[label] += 1
            labels[index] = f"{label} #{seen[label]}"
    return labels


def group_comparisons(results, table):
    if len(results) != len(table):
        raise ValueError("Comparison rows must correspond one-to-one with results")
    from collections import defaultdict
    groups = defaultdict(list)
    for record, row in zip(results, table):
        groups[row["comparison_group"]].append((record, row))
    return dict(groups)


def task_order(results) -> tuple[str, ...]:
    """Use the declared benchmark order, independently of JSON key order."""
    first = results[0]
    tasks = tuple(task["id"] for task in first["config"].get("tasks", ()))
    if not tasks:
        tasks = tuple(first["test"].get("per_task", {}))
    if any(set(record["test"].get("per_task", {})) != set(tasks) for record in results):
        raise ValueError("Per-task metrics differ from the declared task set")
    return tasks


def make_task_tables(results, table):
    """Build percentage tables; saved macro means are never recomputed from rounded cells."""
    tables = []
    for group, pairs in group_comparisons(results, table).items():
        records = [record for record, _ in pairs]
        tasks = task_order(records)
        if not tasks:
            continue
        labels = comparison_labels(records)
        for metric, average in (("top1", "avg_top1"), ("normalized_top1", "avg_normalized_top1")):
            if any(record["test"].get(average) is None or
                   any(record["test"]["per_task"][task].get(metric) is None for task in tasks)
                   for record in records):
                continue
            rows = []
            for record, label in zip(records, labels):
                values = {task: 100 * record["test"]["per_task"][task][metric] for task in tasks}
                rows.append({"Method": label, **values, "Mean": 100 * record["test"][average]})
            tables.append({"group": group, "metric": metric, "model": pairs[0][1]["model"],
                           "benchmark": pairs[0][1]["benchmark"], "rows": rows})
    return tables


def export_task_tables(tables, output_dir):
    """Save readable CSV and Markdown alongside the full-precision comparison CSV."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for table in tables:
        suffix = "accuracy" if table["metric"] == "top1" else "normalized_accuracy"
        stem = f"{table['group'][:12]}_{suffix}_percent"
        rows = [{key: value if key == "Method" else f"{value:.2f}" for key, value in row.items()}
                for row in table["rows"]]
        paths.append(export_csv(rows, output / f"{stem}.csv"))
        title = "Test Accuracy (%)" if table["metric"] == "top1" else "Test Normalized Accuracy (%)"
        columns = list(rows[0])
        lines = [f"# {title}", "", f"{table['model']} / {table['benchmark']}", "",
                 "| " + " | ".join(columns) + " |",
                 "| " + " | ".join("---" if key == "Method" else "---:" for key in columns) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row[key]).replace("|", "\\|") for key in columns) + " |")
        lines += ["", "Mean is the unweighted average across tasks, computed before rounding.",
                  "Values are percentages rounded to two decimal places.", ""]
        path = output / f"{stem}.md"
        with path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        paths.append(path)
    return paths


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
