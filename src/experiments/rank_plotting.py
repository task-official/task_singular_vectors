"""Rank/accuracy plots from recorded spectra and completed evaluation results."""
import math

import numpy as np

from src.experiments.plotting import METHOD_COLORS, _pyplot, _save


def _grid(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#cbd2da", alpha=.8, linewidth=.7)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _short_layer(name):
    suffix = name.removeprefix("model.visual.")
    if suffix == "positional_embedding":
        return "visual.pos"
    if suffix == "proj":
        return "visual.proj"
    suffix = suffix.replace("transformer.resblocks.", "B")
    for old, new in (("attn.in_proj_weight", "QKV"), ("attn.out_proj.weight", "AttnOut"),
                     ("mlp.c_fc.weight", "MLPin"), ("mlp.c_proj.weight", "MLPout")):
        suffix = suffix.replace(old, new)
    return suffix


def _annotate_tasks(figure, axis, x, y, names):
    """Choose nearby label positions with minimal overlap in display pixels."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = []
    candidates = ((7, 7), (7, -15), (-7, 7), (-7, -15), (7, 22), (-7, 22), (15, -28), (-15, -28))
    for xi, yi, name in zip(x, y, names):
        label = axis.annotate(name, (xi, yi), xytext=candidates[0], textcoords="offset points",
                              fontsize=10, arrowprops={"arrowstyle": "-", "color": "#888888", "lw": .5})
        scored = []
        for offset in candidates:
            label.set_position(offset)
            label.set_ha("left" if offset[0] > 0 else "right")
            box = label.get_window_extent(renderer)
            overlap = sum(box.overlaps(previous) for previous in boxes)
            outside = not axis.get_window_extent(renderer).contains(box.x0, box.y0) or not axis.get_window_extent(renderer).contains(box.x1, box.y1)
            scored.append((overlap + 10 * outside, offset))
        _, best = min(scored, key=lambda value: value[0])
        label.set_position(best)
        label.set_ha("left" if best[0] > 0 else "right")
        boxes.append(label.get_window_extent(renderer))


def plot_rank_accuracy(analysis, output, representative_layers):
    plt = _pyplot()
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    tasks = analysis.tasks
    rows = [row for row in analysis.task_rows if row["scope"] == "image_encoder"]
    colors = [METHOD_COLORS[name] for name in ("tsvm", "gse_tsv")]
    title = f"{analysis.sources['tsvm']['config']['model']} / image-encoder matrices"
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    try:
        x = np.arange(len(tasks))
        for offset, method, label, color in zip((-.19, .19), ("tsvm", "gse_tsv"), ("TSV-M", "GSE-TSV"), colors):
            bars = axes[0].bar(x + offset, [row[f"k_{method}"] for row in rows], .36, label=label, color=color)
            axes[0].bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
        axes[0].set_ylabel("Total retained rank (directions)")
        axes[0].set_ylim(0, max(1, max(row[f"k_{method}"] for row in rows for method in ("tsvm", "gse_tsv"))) * 1.2)
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        axes[0].set_title(title)
        bars = axes[1].bar(x, [row["delta_accuracy_pp"] for row in rows], width=.6, color="#526e86", label="GSE-TSV − TSV-M")
        axes[1].bar_label(bars, labels=[f"{row['delta_accuracy_pp']:+.3f}" for row in rows], padding=3)
        axes[1].set_ylabel("Accuracy difference (pp)")
        axes[1].axhline(0, color="#444444", linewidth=.9)
        axes[1].margins(y=.25)
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        axes[1].set_xticks(x, tasks, rotation=20)
        for axis in axes:
            _grid(axis)
        figure.tight_layout()
        paths += _save(figure, output, "task_rank_accuracy")
    finally:
        plt.close(figure)

    from matplotlib.colors import TwoSlopeNorm
    names = [name for name in analysis.layers if name.startswith("model.visual.")]
    lookup = {(row["layer"], row["task"]): row for row in analysis.layer_task_rows}
    values = np.array([[100 * lookup[name, task]["rank_change_fraction"]
                        if lookup[name, task]["rank_change_fraction"] is not None else np.nan
                        for name in names] for task in tasks])
    limit = max(1., float(np.nanmax(np.abs(values))))
    figure, axis = plt.subplots(figsize=(21, 5.3))
    try:
        heatmap = axis.imshow(values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        axis.set_xticks(range(len(names)), [_short_layer(name) for name in names], rotation=90, fontsize=8)
        axis.set_yticks(range(len(tasks)), tasks)
        axis.set_xlabel("Image-encoder layer (matrix); transformer blocks in execution order")
        axis.set_ylabel("Task")
        axis.set_title(title + " — retained-rank change")
        figure.colorbar(heatmap, ax=axis, label="(GSE-TSV rank − TSV-M rank) / TSV-M rank (%)", pad=.015)
        figure.tight_layout()
        paths += _save(figure, output, "layer_rank_change_heatmap")
    finally:
        plt.close(figure)

    for predictor, xlabel, stem in (
        ("rank_change_fraction", "Total retained-rank change (%)", "rank_change_vs_accuracy"),
        ("delta_retained_energy_fraction", "Retained-energy fraction change (pp; energy-weighted across layers)", "energy_change_vs_accuracy"),
    ):
        figure, axis = plt.subplots(figsize=(10, 6))
        try:
            defined = [row for row in rows if row[predictor] is not None]
            x = [100 * row[predictor] for row in defined]
            y = [row["delta_accuracy_pp"] for row in defined]
            axis.scatter(x, y, s=65, color="#1f77b4", edgecolors="white", linewidth=.6, label=f"Tasks (n={len(defined)})")
            axis.axhline(0, color="#777777", linewidth=.8)
            axis.axvline(0, color="#777777", linewidth=.8, linestyle="--")
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Accuracy difference: GSE-TSV − TSV-M (pp)")
            correlation = next(row for row in analysis.correlations if row["group"] == "image_encoder"
                               and row["predictor"] == predictor and row["response"] == "delta_accuracy_pp")
            stats = f"Pearson r={correlation['pearson_r']:.3f}; Spearman ρ={correlation['spearman_rho']:.3f}" if correlation["status"] == "ok" else "Correlation undefined (constant/insufficient observations)"
            axis.set_title(title + "\n" + stats)
            axis.legend(loc="upper left")
            axis.margins(x=.18, y=.2)
            _grid(axis)
            figure.tight_layout()
            _annotate_tasks(figure, axis, x, y, [row["task"] for row in defined])
            paths += _save(figure, output, stem)
        finally:
            plt.close(figure)

    from matplotlib.lines import Line2D
    for name in representative_layers:
        count = math.ceil(len(tasks) / 2)
        figure, axes = plt.subplots(count, 4, figsize=(19, count * 2.8), squeeze=False)
        try:
            for index, task in enumerate(tasks):
                row = lookup[name, task]
                singular = analysis.spectra[name, task]
                ranks = np.arange(1, len(singular) + 1)
                squared = singular ** 2
                cumulative = np.cumsum(squared) / math.fsum(squared) if row["total_energy"] else np.full(len(ranks), np.nan)
                a, b = axes[index // 2, (index % 2) * 2:(index % 2) * 2 + 2]
                a.plot(ranks, np.where(singular > 0, singular, np.nan), color="#333333", linewidth=1.3)
                a.set_yscale("log")
                b.plot(ranks, cumulative, color="#333333", linewidth=1.3)
                b.axhline(.9, color="#aaaaaa", linewidth=.8)
                for method, style, color in zip(("tsvm", "gse_tsv"), ("--", ":"), colors):
                    for axis in (a, b):
                        axis.axvline(row[f"k_{method}"], color=color, linestyle=style, linewidth=1.2)
                for axis in (a, b):
                    if row["r90"] is not None:
                        axis.axvline(row["r90"], color="#777777", linestyle="-.", linewidth=.8)
                    axis.set_xlim(0, len(singular))
                    axis.set_xlabel("Singular-value rank (1-based)", fontsize=8)
                    axis.tick_params(labelsize=8)
                    _grid(axis)
                a.set_ylabel("Singular value σ (log scale)", fontsize=8)
                b.set_ylabel("Cumulative energy fraction", fontsize=8)
                b.set_ylim(0, 1.04)
                a.set_title(f"{task}: E={row['total_energy']:.3g}", fontsize=10)
                b.set_title(f"k: TSV-M={row['k_tsvm']}, GSE-TSV={row['k_gse_tsv']}; r90={row['r90']}", fontsize=9)
            for index in range(len(tasks), count * 2):
                for axis in axes[index // 2, (index % 2) * 2:(index % 2) * 2 + 2]:
                    axis.set_visible(False)
            handles = [Line2D([0], [0], color="#333333", label="Shared input spectrum / cumulative energy"),
                       Line2D([0], [0], color=colors[0], linestyle="--", label="TSV-M retained rank"),
                       Line2D([0], [0], color=colors[1], linestyle=":", label="GSE-TSV retained rank"),
                       Line2D([0], [0], color="#777777", linestyle="-.", label="90% energy rank")]
            figure.suptitle(f"Saved singular spectra — {name}", fontsize=13, y=.995)
            figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .97), ncol=4, fontsize=9)
            figure.tight_layout(rect=(0, 0, 1, .93))
            paths += _save(figure, output, "spectra_" + _short_layer(name).replace(".", "_"))
        finally:
            plt.close(figure)
    return paths
