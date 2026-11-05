"""Relate saved TSV-M/GSE-TSV rank observations to saved task accuracies.

No checkpoints, merge kernels, SVD, datasets or evaluators are used here.
Fractions remain in [0, 1] units in CSV; accuracy differences use percentage
points. A zero-energy spectrum has undefined r90/retained fraction (None).
"""
from dataclasses import dataclass
import csv
import hashlib
import math
from pathlib import Path
import re

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.experiments.analysis import check_comparability, export_csv, method_id, task_order


@dataclass
class RankAccuracyAnalysis:
    tasks: tuple
    layers: tuple
    sources: dict
    layer_task_rows: list
    layer_rows: list
    task_rows: list
    correlations: list
    layer_correlations: list
    spectra: dict


def natural_key(name):
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _mean(values):
    values = [value for value in values if value is not None]
    return math.fsum(values) / len(values) if values else None


def _difference(first, second):
    return second - first if first is not None and second is not None else None


def _display(value, scale=1., digits=3, signed=False):
    if value is None:
        return "undefined"
    return format(scale * value, f"{'+' if signed else ''}.{digits}f")


def _spectrum(layer, task, name):
    values = np.asarray(layer["singular_values"][task], dtype=np.float64)
    q = layer["r"]
    rank = layer["ranks"][task]
    if (values.ndim != 1 or len(values) != q or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(np.diff(values) > 0)):
        raise ValueError(f"{task}/{name}: invalid saved singular spectrum")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= q:
        raise ValueError(f"{task}/{name}: invalid saved rank")
    energy = float(layer["total_energy"][task])
    selected = float(layer["selected_energy_sum"][task])
    for recorded, recomputed in ((energy, math.fsum(values ** 2)),
                                 (selected, math.fsum(values[:rank] ** 2))):
        if not math.isfinite(recorded) or recorded < 0 or not math.isclose(
                recorded, recomputed, rel_tol=2e-6, abs_tol=1e-12):
            raise ValueError(f"{task}/{name}: saved energy disagrees with the spectrum/rank")
    return values, rank, energy, selected


def _accuracy_fields(first, second, task):
    metrics = [record["test"]["per_task"][task] for record in (first, second)]
    result = {}
    for metric, label in (("top1", "accuracy"), ("normalized_top1", "normalized_accuracy")):
        values = [item.get(metric) for item in metrics]
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError(f"{task}: missing/nonfinite saved {metric}")
        result[f"{label}_tsvm_percent"] = 100 * values[0]
        result[f"{label}_gse_tsv_percent"] = 100 * values[1]
        result[f"delta_{label}_pp"] = 100 * (values[1] - values[0])
    return result


def aggregate_rows(rows):
    """Energy-weighted fraction = sum(selected energies) / sum(total energies)."""
    energy = math.fsum(row["total_energy"] for row in rows)
    first = sum(row["k_tsvm"] for row in rows)
    second = sum(row["k_gse_tsv"] for row in rows)
    positive = [row for row in rows if row["total_energy"] > 0]
    result = {"num_matrices": len(rows), "num_positive_energy_matrices": len(positive),
              "k_tsvm": first, "k_gse_tsv": second, "rank_delta": second - first,
              "rank_change_fraction": _ratio(second - first, first),
              "total_energy": energy,
              "r90_sum": sum(row["r90"] for row in positive),
              "r90_sum_over_q_sum": _ratio(sum(row["r90"] for row in positive),
                                             sum(row["q_l"] for row in positive)),
              "mean_layer_r90_over_q": _mean([row["r90_over_q"] for row in positive])}
    for method in ("tsvm", "gse_tsv"):
        selected = math.fsum(row[f"selected_energy_{method}"] for row in rows)
        result[f"selected_energy_{method}"] = selected
        result[f"retained_energy_fraction_{method}"] = _ratio(selected, energy)
        result[f"mean_layer_retained_energy_fraction_{method}"] = _mean(
            [row[f"retained_energy_fraction_{method}"] for row in positive])
    for prefix in ("retained_energy_fraction", "mean_layer_retained_energy_fraction"):
        result[f"delta_{prefix}"] = _difference(result[f"{prefix}_tsvm"], result[f"{prefix}_gse_tsv"])
    return result


def correlation_rows(rows, group, predictors):
    """One observation per TASK, never pool layers as independent accuracies."""
    output = []
    for predictor in predictors:
        for response in ("delta_accuracy_pp", "delta_normalized_accuracy_pp"):
            paired = [(row[predictor], row[response]) for row in rows
                      if row[predictor] is not None and row[response] is not None
                      and math.isfinite(row[predictor]) and math.isfinite(row[response])]
            pearson, spearman = None, None
            status = "insufficient_data"
            if len(paired) >= 2:
                x, y = np.asarray(paired).T
                status = "constant_variable" if np.ptp(x) == 0 or np.ptp(y) == 0 else "ok"
                if status == "ok":
                    pearson, spearman = float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])
            output.append({"group": group, "predictor": predictor, "response": response,
                           "n_tasks": len(paired), "pearson_r": pearson,
                           "spearman_rho": spearman, "status": status})
    return output


def analyze_rank_accuracy(records):
    sources = {}
    for record in records:
        name = method_id(record)
        if name in ("tsvm", "gse_tsv"):
            if name in sources:
                raise ValueError(f"Multiple {name} runs: select the two source runs explicitly")
            sources[name] = record
    if set(sources) != {"tsvm", "gse_tsv"}:
        raise ValueError("One completed TSV-M and one completed GSE-TSV run are required")
    first, second = sources["tsvm"], sources["gse_tsv"]
    check_comparability([first, second])
    tasks = task_order([first, second])
    observations = []
    for record in (first, second):
        diagnostic = record.get("diagnostics", {})
        if diagnostic.get("status") != "collected":
            raise ValueError("Saved rank/spectrum observations were not collected")
        layers = diagnostic.get("observed", {}).get("layers", {})
        if not layers:
            raise ValueError("Saved per-layer spectra are missing; do not recompute a merge")
        observations.append(layers)
    if set(observations[0]) != set(observations[1]):
        raise ValueError("Methods recorded different layer sets")
    layers = tuple(sorted(observations[0], key=natural_key))
    rows, spectra = [], {}
    for index, name in enumerate(layers):
        a, b = (observed[name] for observed in observations)
        for observed in (a, b):
            for key in ("ranks", "singular_values", "total_energy", "selected_energy_sum"):
                if set(observed[key]) != set(tasks):
                    raise ValueError(f"{name}: recorded task sets differ")
            if sum(observed["ranks"].values()) != observed["total_rank"]:
                raise ValueError(f"{name}: total rank disagrees with individual allocations")
        for task in tasks:
            sa, ka, ea, retained_a = _spectrum(a, task, name)
            sb, kb, eb, retained_b = _spectrum(b, task, name)
            if not np.array_equal(sa, sb) or ea != eb:
                raise ValueError(f"{task}/{name}: methods used different saved spectra/energies")
            spectra[name, task] = sa
            squared = sa ** 2
            r90 = None if ea == 0 else min(len(sa), int(np.searchsorted(
                np.cumsum(squared) / math.fsum(squared), .9)) + 1)
            fraction_a, fraction_b = _ratio(retained_a, ea), _ratio(retained_b, ea)
            rows.append({"task": task, "layer": name, "layer_index": index, "q_l": len(sa),
                         "is_image_encoder": name.startswith("model.visual."),
                         "positive_energy": ea > 0,
                         "k_tsvm": ka, "k_gse_tsv": kb, "rank_delta": kb - ka,
                         "rank_change_fraction": _ratio(kb - ka, ka),
                         "total_energy": ea, "r90": r90, "r90_over_q": _ratio(r90, len(sa)) if r90 is not None else None,
                         "selected_energy_tsvm": retained_a, "selected_energy_gse_tsv": retained_b,
                         "retained_energy_fraction_tsvm": fraction_a,
                         "retained_energy_fraction_gse_tsv": fraction_b,
                         "delta_retained_energy_fraction": _difference(fraction_a, fraction_b),
                         **_accuracy_fields(first, second, task)})
    if not any(row["is_image_encoder"] for row in rows):
        raise ValueError("No image-encoder matrix observations were recorded")
    task_rows, correlations = [], []
    for scope in ("image_encoder", "all_recorded_matrices"):
        scoped = []
        for task in tasks:
            chosen = [row for row in rows if row["task"] == task and
                      (scope == "all_recorded_matrices" or row["is_image_encoder"])]
            scoped.append({"scope": scope, "task": task, **aggregate_rows(chosen),
                           **_accuracy_fields(first, second, task)})
        task_rows.extend(scoped)
        correlations.extend(correlation_rows(scoped, scope, (
            "rank_change_fraction", "delta_retained_energy_fraction", "delta_mean_layer_retained_energy_fraction")))
    layer_rows, layer_correlations = [], []
    for name in layers:
        chosen = [row for row in rows if row["layer"] == name]
        layer_rows.append({"layer": name, "q_l": chosen[0]["q_l"],
                           "is_image_encoder": chosen[0]["is_image_encoder"], **aggregate_rows(chosen)})
        layer_correlations.extend(correlation_rows(chosen, name, (
            "rank_change_fraction", "delta_retained_energy_fraction")))
    return RankAccuracyAnalysis(tasks, layers, sources, rows, layer_rows, task_rows,
                                correlations, layer_correlations, spectra)


def spectral_curve_rows(analysis, representative_layers):
    lookup = {(row["layer"], row["task"]): row for row in analysis.layer_task_rows}
    for layer in representative_layers:
        for task in analysis.tasks:
            row = lookup[layer, task]
            singular = analysis.spectra[layer, task]
            squared = singular ** 2
            cumulative = np.cumsum(squared) / math.fsum(squared) if row["total_energy"] else [None] * len(singular)
            for index, (value, energy, fraction) in enumerate(zip(singular, squared, cumulative), 1):
                yield {"layer": layer, "task": task, "component_rank": index,
                       "singular_value": float(value), "squared_singular_value": float(energy),
                       "cumulative_energy_fraction": None if fraction is None else float(fraction),
                       "retained_tsvm": index <= row["k_tsvm"],
                       "retained_gse_tsv": index <= row["k_gse_tsv"]}


def save_rank_analysis(analysis, output_dir, representative_layers):
    """Exclusive report bundle; source records and existing reports stay untouched."""
    from src.experiments.rank_plotting import plot_rank_accuracy

    output = Path(output_dir)
    if not representative_layers or len(set(representative_layers)) != len(representative_layers):
        raise ValueError("Choose nonempty, unique representative layers")
    if any(name not in analysis.layers for name in representative_layers):
        raise ValueError("A representative layer is absent from the saved observations")
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("layer_task_metrics", analysis.layer_task_rows),
                       ("layer_summary", analysis.layer_rows), ("task_summary", analysis.task_rows),
                       ("correlations", analysis.correlations), ("layer_correlations", analysis.layer_correlations)):
        export_csv(rows, output / f"{name}.csv")
    curves = iter(spectral_curve_rows(analysis, representative_layers))
    first = next(curves)
    with (output / "spectral_curves.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(curves)
    plot_rank_accuracy(analysis, output / "figures", representative_layers)
    primary = [row for row in analysis.task_rows if row["scope"] == "image_encoder"]
    tsvm, gse = analysis.sources["tsvm"], analysis.sources["gse_tsv"]
    lines = ["# TSV-M / GSE-TSV：ランク配分と精度の分析", "",
             "保存済みの実験結果と、実際のマージで記録した特異値・配分ランクを使用。新たなマージ・学習・推論は行っていません。", "",
             f"モデル：{tsvm['config']['model']}。タスク数：{len(analysis.tasks)}。",
             f"選択済みα：TSV-M={tsvm['selected_params'].get('alpha')}、GSE-TSV={gse['selected_params'].get('alpha')}。", "",
             "## 集計の定義と確認範囲", "",
             "- 主な図と相関は `image_encoder`：画像側の二次元パラメータのみ。`all_recorded_matrices` は未使用のテキスト側も含めた確認用集計。",
             "- rankは採用した特異方向の数。ゼロ特異値のスロットも保存記録どおり数え、完成した行列の代数的ランクとは区別。",
             "- タスク総ランクは層別ランクの合計。増減率は `(GSE合計−TSV合計)/TSV合計`。分母0は空欄。",
             "- `total_energy` と `selected_energy_*` は保存されたFP32の計算記録を優先。保存された特異値から再計算して整合を確認。",
             "- 主指標の保持エネルギー率は `Σ層の保持エネルギー / Σ層の総エネルギー`。`mean_layer_*` は正エネルギー層の比率の単純平均。",
             "- r90は保存済み特異値からFP64で累積二乗和を求め、90%以上になる最初の順位。元のランク選択処理を再実行した値ではない。",
             "- ゼロエネルギーのr90・保持率は未定義で空欄。単純平均から除外し、エネルギー合算には0として含める。",
             "- CSVの`fraction`は比率（0.1＝10%）。Accuracyの`percent`は百分率、`delta_*_pp`はパーセントポイント。図は増減率を%・保持率差をppで表示。",
             "- 相関は各タスクを1標本として計算。層を独立標本として合算しない。層別相関も各層のタスク間比較。定数・標本不足は未定義。",
             "- 代表層は精度差に基づかず、事前に先頭・中間・最終のAttention QKV行列を指定。両手法の入力特異値は一致している。", "",
             "## タスク別結果（画像エンコーダの全対象層）", "",
             "| Task | TSV-M rank | GSE-TSV rank | Rank change (%) | Retained TSV-M (%) | Retained GSE-TSV (%) | Δ Accuracy (pp) | Δ Normalized Accuracy (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in primary:
        lines.append(f"| {row['task']} | {row['k_tsvm']} | {row['k_gse_tsv']} | {_display(row['rank_change_fraction'], 100, 2, True)} | "
                     f"{_display(row['retained_energy_fraction_tsvm'], 100)} | {_display(row['retained_energy_fraction_gse_tsv'], 100)} | "
                     f"{row['delta_accuracy_pp']:+.4f} | {row['delta_normalized_accuracy_pp']:+.4f} |")
    lines += ["", f"全タスクの総配分数：TSV-M={sum(row['k_tsvm'] for row in primary)}、GSE-TSV={sum(row['k_gse_tsv'] for row in primary)}。",
              f"平均Accuracy差：{100*(gse['test']['avg_top1']-tsvm['test']['avg_top1']):+.6f} pp。平均Normalized Accuracy差：{100*(gse['test']['avg_normalized_top1']-tsvm['test']['avg_normalized_top1']):+.6f} pp。", "",
              "## タスク間相関（画像エンコーダ）", "",
              "| Predictor | Response | n | Pearson r | Spearman ρ |", "|---|---|---:|---:|---:|"]
    for row in analysis.correlations:
        if row["group"] == "image_encoder":
            coefficients = ["undefined" if row[key] is None else f"{row[key]:.6f}" for key in ("pearson_r", "spearman_rho")]
            lines.append(f"| {row['predictor']} | {row['response']} | {row['n_tasks']} | {' | '.join(coefficients)} |")
    increased = [row['task'] for row in primary if row['rank_delta'] > 0 and row['delta_accuracy_pp'] > 0]
    decreased = [row['task'] for row in primary if row['rank_delta'] < 0 and row['delta_accuracy_pp'] < 0]
    opposite = [row['task'] for row in primary if row['rank_delta'] * row['delta_accuracy_pp'] < 0]
    changed_budgets = [row for row in analysis.layer_rows if row['is_image_encoder'] and row['k_tsvm'] != row['k_gse_tsv']]
    lines += ["", "## 観測された関係", "",
              f"- ランク増加とAccuracy改善が同時に見られたタスク：{', '.join(increased) or 'なし'}。",
              f"- ランク減少とAccuracy低下が同時に見られたタスク：{', '.join(decreased) or 'なし'}。",
              f"- ランク増減とAccuracy増減の符号が逆だったタスク：{', '.join(opposite) or 'なし'}。",
              "- 元のタスク差分の保持エネルギーが増えたことと、完成した共通モデルの推論精度が改善することは同義ではありません。", "",
              "## 解釈の限界", "",
              "タスク間の相関は記述的な関係であり、ランク変更が精度差の原因だと証明するものではありません。層別・タスク別のランクを単独で変更した介入実験は行っていません。",
              "保持エネルギーはpolar処理前の元のタスク差分に対する量です。保存していないpolar後の再構成誤差・タスク間干渉を、この量から算出したものとして扱いません。",
              "GSE-TSVは各層で未正規化のσ²上位q個を全タスクから選びます。TSV-Mの均等配分と比べ、タスク差分の大きさとスペクトル形状の両方が配分に影響します。",
              "割り切れない行列では総予算が同じとは限りません。画像側で総配分数が異なる行列：" +
              ("; ".join(f"`{row['layer']}` TSV-M={row['k_tsvm']}, GSE-TSV={row['k_gse_tsv']}" for row in changed_budgets) or "なし") +
              "。未使用のゼロ差分テキスト行列のスロット数の差も確認用集計に残しています。", "",
              "## 入力と再現用の対応", ""]
    for method, record in analysis.sources.items():
        path = Path(record.get("result_path", ""))
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "not_available"
        lines += [f"- {method}: `{path}`", f"  - result file SHA256: `{digest}`",
                  f"  - implementation: `{record['method'].get('implementation', 'not_recorded')}`",
                  f"  - selected weights fingerprint: `{record.get('weights_fingerprint', 'not_recorded')}`"]
    lines += ["", "## 出力", "",
              "- `layer_task_metrics.csv`：全記録行列×タスクの計算値。",
              "- `layer_summary.csv`：行列ごとの全タスク合計。",
              "- `task_summary.csv`：タスクごとの全層集計。2種類のscopeを区別。",
              "- `correlations.csv`：全層集計とAccuracy差・Normalized Accuracy差のPearson/Spearman相関。",
              "- `layer_correlations.csv`：各行列におけるタスク間相関。",
              "- `spectral_curves.csv`：代表層の保存済み特異値・累積エネルギーと両手法の採用範囲。",
              "- `figures/`：PNG・SVG。", ""]
    with (output / "analysis.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output
