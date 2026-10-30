"""Export already-observed diagnostics from saved runs; never recompute a merge."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.analysis import load_results
from src.core.checkpoints import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = load_results(args.runs)
    observed = [{"run": str(Path(record["result_path"]).parent),
                 "recorded_run_dir": record.get("run_dir"),
                 "method": record["method"], "weights_fingerprint": record["weights_fingerprint"],
                 "trace": record.get("trace", {}), "diagnostics": record.get("diagnostics", {"status": "not_collected"})}
                for record in records]
    for item, record in zip(observed, records):
        if record.get("trace_file"):
            path = Path(item["run"]) / record["trace_file"]
            if not path.is_file():
                raise FileNotFoundError(f"Observed trace file missing: {path}")
            expected = record.get("trace_sha256")
            if expected is not None and file_sha256(path) != expected:
                raise ValueError(f"Observed trace checksum mismatch: {path}")
            item["trace_file"] = str(path)
            item["trace_sha256"] = record.get("trace_sha256")
            item["trace_integrity"] = "checksummed" if expected is not None else "not_recorded"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump({"schema_version": 1, "observed_runs": observed}, handle, indent=2)


if __name__ == "__main__":
    main()
