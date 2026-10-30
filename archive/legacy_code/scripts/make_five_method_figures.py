import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsv-merge-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("TSVM", "TSV-M"),
    ("AdaptiveTSV", "Adaptive TSV-M"),
    ("GlobalEnergyTopR", "Global Energy Top-r"),
    ("PerTask95FullConcat", "Per-Task 95% Full Concat"),
    ("FullComponentFullConcat", "Full-Component Full Concat"),
]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsvm",
        default="results/repro_vitb32_8_diagnostics/ViT-B-32_8tasks_TSVM_k=100_nonlinear_additions_svd_diagnostics.json",
    )
    parser.add_argument(
        "--adaptive",
        default="results/repro_vitb32_8_adaptive_rank/ViT-B-32_8tasks_AdaptiveTSV_rank_rho=0.95_k=100_nonlinear_additions_.json",
    )
    parser.add_argument(
        "--global-energy",
        default="results/five_method_comparison/ViT-B-32_8tasks_GlobalEnergyTopR_k=100_nonlinear_additions_.json",
    )
    parser.add_argument(
        "--per-task-95",
        default="results/five_method_comparison/ViT-B-32_8tasks_PerTask95FullConcat_rho=0.95_k=100_nonlinear_additions_.json",
    )
    parser.add_argument(
        "--full-component",
        default="results/five_method_comparison/ViT-B-32_8tasks_FullComponentFullConcat_k=100_nonlinear_additions_.json",
    )
    parser.add_argument(
        "--diagnostics",
        default="results/five_method_comparison/ViT-B-32_8tasks_merge_diagnostics.json",
    )
    parser.add_argument("--output-dir", default="five_method_figures")
    return parser.parse_args()


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, output_dir, stem):
    for extension in ("png", "svg"):
        directory = output_dir / extension
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / f"{stem}.{extension}", bbox_inches="tight", dpi=220)
    plt.close(figure)


def style_axes(axis, grid_axis="y"):
    axis.grid(axis=grid_axis, alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def annotate_heatmap(axis, values, decimals=2):
    finite = values[np.isfinite(values)]
    threshold = np.max(np.abs(finite)) * 0.55 if finite.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            color = "white" if abs(value) > threshold else "black"
            axis.text(
                column,
                row,
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                fontsize=5,
                color=color,
            )


def task_names(result):
    return [
        key.removesuffix(":top1")
        for key in result["test"]
        if key.endswith(":top1") and not key.startswith("avg_")
    ]


def test_accuracy(result, task):
    return float(result["test"][f"{task}:top1"])


def average_metric(result, name, suffix):
    if name in result["test"]:
        return float(result["test"][name])
    values = [
        float(value)
        for key, value in result["test"].items()
        if key.endswith(suffix) and not key.startswith("avg_")
    ]
    return float(np.mean(values))


def diagnostic_layers(diagnostics):
    return list(diagnostics["rank"]["per_layer"])


def method_diagnostics(artifact, method):
    diagnostics = artifact["methods"].get(method)
    if diagnostics is None:
        raise KeyError(f"Diagnostics are missing for {method}")
    return diagnostics


def figure_1(results, labels, output_dir):
    absolute = [average_metric(result, "avg_top1", ":top1") for result in results]
    normalized = [
        average_metric(result, "avg_normalized_top1", ":normalized_top1")
        for result in results
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axis, values, title in zip(
        axes,
        (absolute, normalized),
        ("Average Absolute Test Accuracy", "Average Normalized Test Accuracy"),
    ):
        bars = axis.bar(range(len(labels)), values, color=COLORS)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.set_ylabel("Accuracy")
        axis.set_title(title)
        style_axes(axis)
    rows = [
        {"method": label, "average_absolute_accuracy": abs_value, "average_normalized_accuracy": norm_value}
        for label, abs_value, norm_value in zip(labels, absolute, normalized)
    ]
    write_csv(output_dir / "data/01_average_accuracy.csv", list(rows[0]), rows)
    figure.suptitle("ViT-B-32, 8 Tasks")
    save_figure(figure, output_dir, "01_average_accuracy")


def figure_2(results, tasks, labels, output_dir):
    baseline = np.array([test_accuracy(results[0], task) for task in tasks])
    values = np.array(
        [[100 * (test_accuracy(result, task) - base) for result in results] for task, base in zip(tasks, baseline)]
    )
    limit = max(0.1, float(np.max(np.abs(values))))
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    annotate_heatmap(axis, values, decimals=2)
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(range(len(tasks)), tasks)
    axis.set_title("Absolute Test Accuracy Difference from TSV-M")
    figure.colorbar(image, ax=axis, label="Difference (percentage points)")
    rows = [
        {"task": task, "method": label, "difference_percentage_points": values[i, j]}
        for i, task in enumerate(tasks)
        for j, label in enumerate(labels)
    ]
    write_csv(output_dir / "data/02_task_accuracy_difference.csv", list(rows[0]), rows)
    save_figure(figure, output_dir, "02_task_accuracy_difference_heatmap")


def figure_3(results, tasks, labels, output_dir):
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    figure, axis = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    rows = []
    for result, label, color in zip(results, labels, COLORS):
        values = np.array([100 * test_accuracy(result, task) for task in tasks])
        axis.plot(closed_angles, np.append(values, values[0]), label=label, color=color, linewidth=2)
        for task, value in zip(tasks, values):
            rows.append({"method": label, "task": task, "absolute_accuracy_percent": value})
    axis.set_xticks(angles, tasks)
    axis.set_ylabel("Absolute accuracy (%)")
    axis.set_title("Task-wise Absolute Test Accuracy", pad=25)
    axis.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))
    write_csv(output_dir / "data/03_task_absolute_accuracy.csv", list(rows[0]), rows)
    save_figure(figure, output_dir, "03_task_absolute_accuracy_radar")


def figure_4(results, method_keys, labels, artifact, output_dir):
    widths = [method_diagnostics(artifact, method)["component_width"]["normalized_width"] for method in method_keys]
    accuracies = [average_metric(result, "avg_normalized_top1", ":normalized_top1") for result in results]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for width, accuracy, label, color in zip(widths, accuracies, labels, COLORS):
        axis.scatter(width, accuracy, s=80, color=color)
        axis.annotate(label, (width, accuracy), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel(r"Normalized concatenated width $Q_m=\sum_l K_{m,l}/\sum_l r_l$")
    axis.set_ylabel("Average normalized test accuracy")
    axis.set_title("Accuracy vs. Concatenated Component Width")
    style_axes(axis, "both")
    rows = [{"method": label, "normalized_component_width": width, "average_normalized_accuracy": accuracy} for label, width, accuracy in zip(labels, widths, accuracies)]
    write_csv(output_dir / "data/04_accuracy_vs_width.csv", list(rows[0]), rows)
    save_figure(figure, output_dir, "04_accuracy_vs_component_width")


def figure_5(method_keys, labels, artifact, layers, output_dir):
    values = np.array([
        [method_diagnostics(artifact, method)["rank"]["per_layer"][layer]["total_rank"] / method_diagnostics(artifact, method)["rank"]["per_layer"][layer]["r"] for layer in layers]
        for method in method_keys
    ])
    figure, axis = plt.subplots(figsize=(max(18, len(layers) * 0.42), 4.5))
    image = axis.imshow(values, cmap="viridis", aspect="auto")
    annotate_heatmap(axis, values, decimals=2)
    axis.set_xticks(range(len(layers)), layers, rotation=90, fontsize=6)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title(r"Layer-wise Concatenated Width $K_{m,l}/r_l$")
    figure.colorbar(image, ax=axis, label="Width ratio")
    rows = [{"method": label, "layer": layer, "width_ratio": values[i, j]} for i, label in enumerate(labels) for j, layer in enumerate(layers)]
    write_csv(output_dir / "data/05_layer_width_ratio.csv", list(rows[0]), rows)
    save_figure(figure, output_dir, "05_layer_width_heatmap")


def figure_6(artifact, tasks, layers, output_dir):
    methods = [("AdaptiveTSV", "Adaptive TSV-M"), ("GlobalEnergyTopR", "Global Energy Top-r")]
    figure, axes = plt.subplots(2, 1, figsize=(max(18, len(layers) * 0.42), 9), constrained_layout=True)
    rows = []
    for axis, (method, label) in zip(axes, methods):
        diagnostics = method_diagnostics(artifact, method)
        values = np.array([
            [
                diagnostics["rank"]["per_layer"][layer]["ranks"][task]
                / diagnostics["rank"]["per_layer"][layer]["base_rank"]
                if diagnostics["rank"]["per_layer"][layer]["base_rank"] > 0
                else np.nan
                for layer in layers
            ]
            for task in tasks
        ])
        image = axis.imshow(values, cmap="magma", aspect="auto", vmin=0)
        axis.set_xticks(range(len(layers)), layers, rotation=90, fontsize=6)
        axis.set_yticks(range(len(tasks)), tasks)
        axis.set_title(label)
        figure.colorbar(image, ax=axis, label=r"$k_{m,i}^{(l)}/k_{base}^{(l)}$")
        rows.extend({"method": label, "task": task, "layer": layer, "rank_ratio": values[i, j]} for i, task in enumerate(tasks) for j, layer in enumerate(layers))
    write_csv(output_dir / "data/06_task_layer_rank_ratio.csv", list(rows[0]), rows)
    save_figure(figure, output_dir, "06_task_layer_rank_allocation_heatmaps")


def figure_7(method_keys, labels, artifact, tasks, output_dir):
    values = [[method_diagnostics(artifact, method)["selection_energy"]["task_scores"][task] for task in tasks] for method in method_keys]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.boxplot(values, showfliers=False)
    rows = []
    for method_index, (method_values, label, color) in enumerate(zip(values, labels, COLORS), start=1):
        offsets = np.linspace(-0.12, 0.12, len(method_values))
        axis.scatter(method_index + offsets, method_values, color=color, s=30, zorder=3)
        rows.extend({"method": label, "task": task, "selection_energy": value} for task, value in zip(tasks, method_values))
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("Cross-layer selection energy")
    axis.set_title("Task-wise Selection Energy Distribution")
    axis.set_ylim(bottom=0)
    style_axes(axis)
    write_csv(output_dir / "data/07_selection_energy.csv", list(rows[0]), rows)
    save_figure(figure, output_dir, "07_selection_energy_distribution")


def figure_8(results, method_keys, labels, artifact, tasks, output_dir):
    baseline_energy = method_diagnostics(artifact, "TSVM")["selection_energy"]["task_scores"]
    baseline_accuracy = {task: test_accuracy(results[0], task) for task in tasks}
    figure, axis = plt.subplots(figsize=(8, 6))
    rows = []
    for result, method, label, color in zip(results[1:], method_keys[1:], labels[1:], COLORS[1:]):
        energy = method_diagnostics(artifact, method)["selection_energy"]["task_scores"]
        x_values = [energy[task] - baseline_energy[task] for task in tasks]
        y_values = [100 * (test_accuracy(result, task) - baseline_accuracy[task]) for task in tasks]
        axis.scatter(x_values, y_values, label=label, color=color, alpha=0.8)
        rows.extend({"method": label, "task": task, "selection_energy_delta": x, "accuracy_delta_percentage_points": y} for task, x, y in zip(tasks, x_values, y_values))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Selection energy change from TSV-M")
    axis.set_ylabel("Absolute accuracy change (percentage points)")
    axis.set_title("Selection Energy Change vs. Accuracy Change")
    axis.legend()
    style_axes(axis, "both")
    write_csv(output_dir / "data/08_energy_vs_accuracy_delta.csv", list(rows[0]), rows)
    save_figure(figure, output_dir, "08_selection_energy_vs_accuracy_change")


def figure_9(method_keys, labels, artifact, layers, output_dir):
    figure, axis = plt.subplots(figsize=(9, 6))
    rows = []
    for method, label, color in zip(method_keys, labels, COLORS):
        per_layer = method_diagnostics(artifact, method)["procrustes"]["per_layer"]
        x_values, y_values = [], []
        for layer in layers:
            deformation = per_layer[layer]["deformation_rate"]
            if deformation is None:
                continue
            width = per_layer[layer]["width_ratio"]
            x_values.append(width)
            y_values.append(deformation)
            rows.append({"method": label, "layer": layer, "width_ratio": width, "procrustes_deformation": deformation})
        axis.scatter(x_values, y_values, label=label, color=color, alpha=0.65, s=24)
    axis.set_xlabel(r"Layer width ratio $K_{m,l}/r_l$")
    axis.set_ylabel("Procrustes deformation rate")
    axis.set_title("Polar-Factor Deformation vs. Concatenated Width")
    axis.legend()
    style_axes(axis, "both")
    write_csv(output_dir / "data/09_procrustes_deformation.csv", list(rows[0]), rows)
    save_figure(figure, output_dir, "09_procrustes_deformation_vs_width")


def resource_figure(results, method_keys, labels, artifact, output_dir, metric, stem, xlabel, scale=1.0):
    points = []
    for result, method, label, color in zip(results, method_keys, labels, COLORS):
        value = method_diagnostics(artifact, method)["performance"].get(metric)
        if value is None:
            continue
        accuracy = average_metric(result, "avg_normalized_top1", ":normalized_top1")
        points.append((float(value) / scale, accuracy, label, color))
    if not points:
        print(f"Skipping {stem}: {metric} is unavailable.")
        return
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for value, accuracy, label, color in points:
        axis.scatter(value, accuracy, s=80, color=color)
        axis.annotate(label, (value, accuracy), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Average normalized test accuracy")
    axis.set_title(xlabel + " vs. Accuracy")
    style_axes(axis, "both")
    rows = [
        {
            "method": label,
            "resource_value_in_display_units": value,
            "average_normalized_accuracy": accuracy,
        }
        for value, accuracy, label, _ in points
    ]
    write_csv(output_dir / f"data/{stem}.csv", list(rows[0]), rows)
    save_figure(figure, output_dir, stem)


def main():
    args = parse_args()
    result_paths = [args.tsvm, args.adaptive, args.global_energy, args.per_task_95, args.full_component]
    results = [load_json(path) for path in result_paths]
    artifact = load_json(args.diagnostics)
    output_dir = Path(args.output_dir)
    method_keys = [method for method, _ in METHODS]
    labels = [label for _, label in METHODS]
    tasks = task_names(results[0])
    layers = diagnostic_layers(method_diagnostics(artifact, "TSVM"))

    figure_1(results, labels, output_dir)
    figure_2(results, tasks, labels, output_dir)
    figure_3(results, tasks, labels, output_dir)
    figure_4(results, method_keys, labels, artifact, output_dir)
    figure_5(method_keys, labels, artifact, layers, output_dir)
    figure_6(artifact, tasks, layers, output_dir)
    figure_7(method_keys, labels, artifact, tasks, output_dir)
    figure_8(results, method_keys, labels, artifact, tasks, output_dir)
    figure_9(method_keys, labels, artifact, layers, output_dir)
    resource_figure(
        results,
        method_keys,
        labels,
        artifact,
        output_dir,
        "peak_gpu_memory_bytes",
        "10_accuracy_vs_peak_gpu_memory",
        "Peak GPU memory (GiB)",
        scale=1024**3,
    )
    resource_figure(
        results,
        method_keys,
        labels,
        artifact,
        output_dir,
        "merge_seconds",
        "11_accuracy_vs_merge_time",
        "Merge time (seconds)",
    )
    print(f"Saved figures and CSV files under {output_dir}")


if __name__ == "__main__":
    main()
