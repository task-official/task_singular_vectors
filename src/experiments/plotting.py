"""Figures derived only from saved comparison records; no model/data imports."""
from pathlib import Path


def plot_comparison(results, table, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for record, row in zip(results, table):
        groups[row["comparison_group"]].append((record, row))
    paths = []
    for group, pairs in groups.items():
        names = [row["method"] for _, row in pairs]
        if len(set(names)) != len(names):
            import json
            names = [f"{row['method']} [{row['implementation']}] {json.dumps(record.get('selected_params', {}), sort_keys=True)}"
                     for record, row in pairs]
            counts, seen = Counter(names), Counter()
            for index, label in enumerate(names):
                if counts[label] > 1:
                    seen[label] += 1
                    names[index] = f"{label} #{seen[label]}"
        for key in ("avg_top1", "avg_normalized_top1"):
            values = [row[key] for _, row in pairs]
            if any(value is None for value in values):
                continue
            figure, axis = plt.subplots(figsize=(max(6, len(names)*1.2), 4))
            axis.bar(range(len(names)), [100*value for value in values])
            axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
            axis.set_ylabel("Accuracy (%)" if key == "avg_top1" else "Normalized accuracy (%)")
            axis.set_title(f"{pairs[0][1]['model']} / {pairs[0][1]['benchmark']} / {group[:8]}")
            axis.set_ylim(bottom=0)  # Normalized values above 100 are preserved.
            figure.tight_layout()
            for extension in ("png", "svg"):
                path = output / f"{group[:12]}_{key}.{extension}"
                figure.savefig(path)
                paths.append(str(path))
            plt.close(figure)
        tasks = tuple(pairs[0][0]["test"].get("per_task", {}))
        if tasks and all(tuple(record["test"].get("per_task", {})) == tasks for record, _ in pairs):
            figure, axis = plt.subplots(figsize=(max(8, len(tasks)*.7), 4))
            for index, (record, row) in enumerate(pairs):
                axis.plot(tasks, [100*record["test"]["per_task"][task]["top1"] for task in tasks], marker="o", label=names[index])
            axis.set_ylabel("Accuracy (%)")
            axis.tick_params(axis="x", rotation=45)
            axis.legend()
            figure.tight_layout()
            for extension in ("png", "svg"):
                path = output / f"{group[:12]}_task_accuracy.{extension}"
                figure.savefig(path)
                paths.append(str(path))
            plt.close(figure)
    return paths
