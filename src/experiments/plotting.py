"""Figures derived only from saved results; no merging, model or data loading."""
import hashlib
import math
from pathlib import Path

from src.experiments.analysis import comparison_labels, group_comparisons, method_id, task_order


# Presentation defaults only: an unfamiliar method still plots without registration.
METHOD_COLORS = {
    "zeroshot": "#7f7f7f", "weight_average": "#bcbd22", "task_arithmetic": "#17becf",
    "ties": "#e377c2", "consensus_ta": "#8c564b", "tsvm": "#1f77b4",
    "iso_cts": "#ff7f0e", "dc_merge": "#2ca02c", "gse_tsv": "#d62728", "sera": "#9467bd",
}
METHOD_MARKERS = {"tsvm": "o", "iso_cts": "^", "dc_merge": "s", "gse_tsv": "v", "sera": "D"}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(figure, output, stem):
    paths = [output / f"{stem}.{extension}" for extension in ("png", "svg")]
    for path in paths:
        if path.exists():
            raise FileExistsError(path)
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return [str(path) for path in paths]


def _style_axis(axis, step):
    from matplotlib.ticker import MultipleLocator
    axis.set_axisbelow(True)
    axis.yaxis.set_major_locator(MultipleLocator(step))
    axis.grid(axis="y", color="#cbd2da", linewidth=.7, alpha=.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _series_style(name):
    palette = tuple(METHOD_COLORS.values())
    fallback = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "big")
    style_index = tuple(METHOD_COLORS).index(name) if name in METHOD_COLORS else fallback
    return {"color": METHOD_COLORS.get(name, palette[fallback % len(palette)]),
            "marker": METHOD_MARKERS.get(name, ("o", "s", "^", "D", "v")[style_index % 5]),
            "linestyle": ("-", "--", "-.", ":")[style_index % 4]}


def plot_comparison(results, table, output_dir, *, task_metric="accuracy"):
    """Keep absolute/normalized task metrics explicit, including values above 100%."""
    if task_metric not in ("accuracy", "normalized_accuracy", "both"):
        raise ValueError(f"Unknown task metric: {task_metric}")
    plt = _pyplot()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for group, pairs in group_comparisons(results, table).items():
        records = [record for record, _ in pairs]
        names = comparison_labels(records)
        colors = [_series_style(method_id(record))["color"] for record in records]
        title = f"{pairs[0][1]['model']} / {pairs[0][1]['benchmark']}"
        for key in ("avg_top1", "avg_normalized_top1"):
            values = [row[key] for _, row in pairs]
            if any(value is None for value in values):
                continue
            percentages = [100 * value for value in values]
            figure, axis = plt.subplots(figsize=(max(8, len(names) * 1.15), 5))
            try:
                bars = axis.bar(range(len(names)), percentages, color=colors, width=.67)
                axis.bar_label(bars, labels=[f"{value:.2f}" for value in percentages], padding=4, fontsize=10)
                axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
                normalized = key == "avg_normalized_top1"
                axis.set_ylabel("Normalized Accuracy (%)" if normalized else "Accuracy (%)")
                axis.set_title(title)
                # Include a labelled 100 tick without clipping normalized scores >100.
                axis.set_ylim(0, 10 * math.ceil(max(100, max(percentages) + 5) / 10))
                _style_axis(axis, 20)
                if normalized:
                    axis.axhline(100, color="#606873", linewidth=1, linestyle="--")
                figure.tight_layout()
                paths += _save(figure, output, f"{group[:12]}_{key}")
            finally:
                plt.close(figure)
        tasks = task_order(records)
        if not tasks:
            continue
        metrics = []
        if task_metric in ("accuracy", "both"):
            metrics.append(("top1", "task_accuracy", "Accuracy (%)"))
        if task_metric in ("normalized_accuracy", "both"):
            metrics.append(("normalized_top1", "task_normalized_accuracy", "Normalized Accuracy (%)"))
        for metric, suffix, ylabel in metrics:
            values = [[record["test"]["per_task"][task].get(metric) for task in tasks] for record in records]
            if any(value is None for row in values for value in row):
                if task_metric == "normalized_accuracy":
                    raise ValueError("Per-task normalized accuracy was not recorded; cannot relabel absolute values")
                continue
            percentages = [[100 * value for value in row] for row in values]
            figure, axis = plt.subplots(figsize=(max(11, len(tasks) * 1.05 + 3), 5.5))
            try:
                for index, (record, series) in enumerate(zip(records, percentages)):
                    axis.plot(tasks, series, label=names[index], linewidth=1.8, markersize=5,
                              **_series_style(method_id(record)))
                flat = [value for row in percentages for value in row]
                lower = max(0, 5 * math.floor((min(flat) - 3) / 5))
                upper = 5 * math.ceil(max(100, max(flat) + 3) / 5)
                axis.set_ylim(lower, upper)
                _style_axis(axis, 20 if upper - lower > 60 else 5)
                if metric == "normalized_top1":
                    axis.axhline(100, color="#606873", linewidth=1, linestyle="--")
                axis.set_ylabel(ylabel)
                axis.set_title(title)
                axis.tick_params(axis="x", rotation=30)
                axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)
                figure.tight_layout(rect=(0, 0, .8, 1))
                paths += _save(figure, output, f"{group[:12]}_{suffix}")
            finally:
                plt.close(figure)
    return paths


def plot_task_tables(tables, output_dir):
    """Render the same percentage rows exported as CSV/Markdown, without re-evaluation."""
    plt = _pyplot()
    output = Path(output_dir)
    paths = []
    for table in tables:
        rows = table["rows"]
        columns = list(rows[0])
        cells = [[row[key] if key == "Method" else f"{row[key]:.2f}" for key in columns] for row in rows]
        figure, axis = plt.subplots(figsize=(max(12, len(columns) * 1.1 + 2), len(rows) * .4 + 1.4))
        try:
            axis.axis("off")
            normalized = table["metric"] == "normalized_top1"
            title = "Test Normalized Accuracy (%)" if normalized else "Test Accuracy (%)"
            axis.set_title(f"{title} — {table['model']} / {table['benchmark']}", loc="left", pad=14)
            grid = axis.table(cellText=cells, colLabels=columns, cellLoc="right", loc="center",
                              colWidths=[.24] + [.76 / (len(columns) - 1)] * (len(columns) - 1),
                              bbox=[0, .07, 1, .85])
            grid.auto_set_font_size(False)
            grid.set_fontsize(9.5)
            for (row, column), cell in grid.get_celld().items():
                cell.set_edgecolor("#d5dce3")
                cell.set_linewidth(.5)
                if row == 0:
                    cell.set_facecolor("#29465b")
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                else:
                    cell.set_facecolor("#f4f7fa" if row % 2 else "white")
                    if column == len(columns) - 1:
                        cell.set_facecolor("#e8eff6")
                        cell.get_text().set_weight("bold")
                if column == 0:
                    cell.get_text().set_ha("left")
            figure.text(.02, .025, "Mean: unweighted average across tasks; values rounded to two decimal places.", fontsize=8)
            figure.tight_layout()
            suffix = "normalized_accuracy" if normalized else "accuracy"
            paths += _save(figure, output, f"{table['group'][:12]}_{suffix}_percent")
        finally:
            plt.close(figure)
    return paths
