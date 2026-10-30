"""Read historical evidence without modifying it or inferring provenance."""
import json
from pathlib import Path


def read_legacy_result(path):
    source = Path(path)
    result = json.loads(source.read_text())
    if "test" not in result:
        raise ValueError("Expected a legacy result containing test metrics")
    return {"source": str(source), "provenance_status": "historical_unverified",
            "comparable_to_new_runs": False, "legacy": result}


def export_legacy_result(result, destination):
    """Explicit export only; never overwrite a historical result."""
    def flatten(metrics):
        output = {"avg_top1": metrics["avg_top1"], "avg_normalized_top1": metrics["avg_normalized_top1"]}
        for task, values in metrics["per_task"].items():
            output[f"{task}:top1"] = values["top1"]
            output[f"{task}:normalized_top1"] = values["normalized_top1"]
        return output
    with Path(destination).open("x") as handle:
        json.dump({"test": flatten(result["test"]),
                   "selected_params": result["selected_params"],
                   "source_run": result["run_dir"]}, handle, indent=2)


def legacy_view(path, label="historical_unknown"):
    """Explicitly label unknown provenance; never infer conditions from a filename."""
    evidence = read_legacy_result(path)
    raw = evidence["legacy"]
    metrics = raw["test"]
    per_task = {}
    for key, value in metrics.items():
        if key.endswith(":top1"):
            task = key[:-5]
            per_task[task] = {"top1": value, "normalized_top1": metrics.get(f"{task}:normalized_top1")}
    return {"schema_version": 1, "status": "complete", "source_type": "historical_unverified",
            "config": {"model": "unknown", "benchmark": "unknown", "protocol": "historical_unverified"},
            "input_identity": {"unverified_source": str(path)},
            "method": {"id": label, "implementation": "unknown"},
            "selected_params": raw.get("selected_params", {}), "weights_fingerprint": "unknown",
            "test": {"avg_top1": metrics.get("avg_top1"),
                     "avg_normalized_top1": metrics.get("avg_normalized_top1"), "per_task": per_task},
            "result_path": str(path)}
