import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Style ──────────────────────────────────────────────────────────────────────

PALETTE = {
    "blue": "#378ADD",
    "darker-blue": "#0057B8",
    "darker-green": "#007A4D",
    "teal": "#1D9E75",
    "coral": "#D85A30",
    "amber": "#BA7517",
    "purple": "#7F77DD",
    "gray": "#888780",
}
COLORS = list(PALETTE.values())

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
})


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_delta_results(input_dir, topk=128):
    paths = sorted(glob.glob(os.path.join(input_dir, "**", f"*k{topk}.*.json"), recursive=True))
    rows = []
    for path in paths:
        obj = load_json(path)
        if "pca_summary" not in obj or "mean_delta_summary" not in obj:
            continue
        cfg = obj.get("config", {})
        row = {
            "path": path,
            "file": os.path.basename(path),
            "model": cfg.get("model"),
            "train_data_prefix": cfg.get("train_data_prefix"),
            "eval_data_prefix": cfg.get("eval_data_prefix"),
            "activation_type": cfg.get("activation_type"),
            "layer": cfg.get("layer"),
            "layer_num": cfg.get("layer_num"),
            "agg_type": cfg.get("agg_type"),
            "pca_n_components": cfg.get("pca_n_components"),
            "center_eval": cfg.get("center_eval"),
            "num_train": obj.get("num_train"),
            "num_eval": obj.get("num_eval"),
        }
        for k, v in obj["pca_summary"].items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        for k, v in obj["mean_delta_summary"].items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        rows.append(row)

    if not rows:
        raise ValueError(f"No delta comparison JSON files found under {input_dir}")
    return pd.DataFrame(rows)


def load_per_sample_delta_results(input_dir, topk=128):
    paths = sorted(glob.glob(os.path.join(input_dir, "**", f"*k{topk}.*.json"), recursive=True))
    rows = []

    for path in paths:
        obj = load_json(path)
        if "per_example" not in obj or "config" not in obj:
            continue

        cfg = obj["config"]

        for ex in obj["per_example"]:
            row = {
                "path": path,
                "file": os.path.basename(path),
                "model": cfg.get("model"),
                "train_data_prefix": cfg.get("train_data_prefix"),
                "eval_data_prefix": cfg.get("eval_data_prefix"),
                "activation_type": cfg.get("activation_type"),
                "layer": cfg.get("layer"),
                "layer_num": cfg.get("layer_num"),
                "agg_type": cfg.get("agg_type"),
                "pca_n_components": cfg.get("pca_n_components"),
                "center_eval": cfg.get("center_eval"),
                "question_id": ex.get("question_id"),
            }
            for k, v in ex.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[k] = v
            rows.append(row)

    if not rows:
        raise ValueError(f"No per-example rows found under {input_dir}")
    return pd.DataFrame(rows)


def add_metrics(df):
    df = df.copy()
    if {"mean_pca_cosine_reconstructed_eval_delta",
        "mean_cosine_eval_delta_train_mean_delta"}.issubset(df.columns):
        df["pca_vs_mean_cosine_gain"] = (
                df["mean_pca_cosine_reconstructed_eval_delta"]
                - df["mean_cosine_eval_delta_train_mean_delta"]
        )
    if {"mean_pca_residual_fraction",
        "mean_pca_projection_fraction"}.issubset(df.columns):
        df["pca_projection_minus_residual_fraction"] = (
                df["mean_pca_projection_fraction"]
                - df["mean_pca_residual_fraction"]
        )
    return df


def short_label(row):
    return " | ".join([
        str(row.get("activation_type", "")),
        str(row.get("layer", "")),
        str(row.get("layer_num", "")),
        str(row.get("agg_type", "")),
    ])


def full_label(row):
    """Two-line label: config info on line 1, model/data context on line 2."""
    line1 = " | ".join([
        str(row.get("activation_type", "")),
        str(row.get("layer", "")),
        str(row.get("layer_num", "")),
        str(row.get("agg_type", "")),
    ])
    model = str(row.get("model", "") or "")
    train = str(row.get("train_data_prefix", "") or "")
    evl = str(row.get("eval_data_prefix", "") or "")
    # Shorten long path-like prefixes to the final component
    train = train.split("/")[-1] if "/" in train else train
    evl = evl.split("/")[-1] if "/" in evl else evl
    line2 = f"{model}  tr:{train}  ev:{evl}"
    return f"{line1}\n{line2}"


def save(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ── Plot 1: Multi-metric heatmap ───────────────────────────────────────────────

def plot_heatmap(df, output_dir, top_k=None):
    """
    Rows = all configurations (sorted by mean-delta cosine), or top_k if given.
    Columns = key metrics, each column z-scored so colors are comparable.
    """
    metrics = [
        "mean_cosine_eval_delta_train_mean_delta",
        "mean_pca_cosine_reconstructed_eval_delta",
        "pca_vs_mean_cosine_gain",
        "pca_train_cumulative_explained_variance_ratio",
        "mean_eval_delta_norm",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return

    sort_col = "mean_cosine_eval_delta_train_mean_delta"
    if sort_col not in df.columns:
        sort_col = metrics[0]

    plot_df = (
        df.dropna(subset=metrics)
        .copy()
        .sort_values(sort_col, ascending=False)
    )
    if top_k is not None:
        plot_df = plot_df.head(top_k)
    if plot_df.empty:
        return

    plot_df["label"] = plot_df.apply(full_label, axis=1)

    matrix = plot_df[metrics].values.astype(float)
    col_mean = np.nanmean(matrix, axis=0)
    col_std = np.nanstd(matrix, axis=0)
    col_std[col_std == 0] = 1
    z = (matrix - col_mean) / col_std

    short_names = [m.replace("mean_", "").replace("pca_", "pca·")
                   .replace("_", " ") for m in metrics]

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.4),
                                    max(6, n * 0.55)))
    im = ax.imshow(z, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_df["label"], fontsize=7.5, linespacing=1.4)
    title_desc = f"top {top_k}" if top_k is not None else f"all {n}"
    ax.set_title(f"Multi-metric heatmap — {title_desc} configs sorted by mean-delta cosine\n"
                 "(colors are z-scored per column; green = better)", fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="z-score")
    fig.tight_layout()
    fname = f"heatmap_top{top_k}_configs.pdf" if top_k is not None else "heatmap_all_configs.pdf"
    save(fig, output_dir, fname)


# ── Plot 2: Bubble scatter ─────────────────────────────────────────────────────

def plot_bubble_scatter(df, output_dir):
    x_col = "mean_cosine_eval_delta_train_mean_delta"
    y_col = "mean_pca_cosine_reconstructed_eval_delta"
    s_col = "pca_train_cumulative_explained_variance_ratio"
    c_col = "agg_type"

    need = [x_col, y_col]
    if not all(c in df.columns for c in need):
        return

    plot_df = df.dropna(subset=need).copy()
    if plot_df.empty:
        return

    agg_types = plot_df[c_col].unique() if c_col in plot_df.columns else ["all"]
    color_map = {a: COLORS[i % len(COLORS)] for i, a in enumerate(agg_types)}

    fig, ax = plt.subplots(figsize=(7, 7))

    for agg, grp in (plot_df.groupby(c_col) if c_col in plot_df.columns
    else [("all", plot_df)]):
        sizes = (
            (grp[s_col].fillna(0.5) * 300).clip(20, 600)
            if s_col in grp.columns else 80
        )
        ax.scatter(grp[x_col], grp[y_col],
                   s=sizes, c=color_map[agg], alpha=0.65,
                   edgecolors="white", linewidths=0.5, label=str(agg))

    lo = min(plot_df[x_col].min(), plot_df[y_col].min()) - 0.02
    hi = max(plot_df[x_col].max(), plot_df[y_col].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="#AAAAAA", linewidth=1, label="y = x")

    ax.set_xlabel("Mean-delta cosine with eval deltas")
    ax.set_ylabel("PCA reconstruction cosine with eval deltas")
    ax.set_title("PCA subspace vs mean delta\n"
                 "(bubble size = PCA explained variance, color = agg type)")
    ax.legend(fontsize=9, title=c_col, title_fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    save(fig, output_dir, "bubble_pca_vs_mean_delta.pdf")


# ── Plot 4: PCA gain breakdown (strip / swarm style) ──────────────────────────

def plot_gain_by_group(df, output_dir):
    gain_col = "pca_vs_mean_cosine_gain"
    if gain_col not in df.columns:
        return

    group_col = next((c for c in ["activation_type", "agg_type", "layer"]
                      if c in df.columns), None)
    if group_col is None:
        return

    plot_df = df.dropna(subset=[gain_col, group_col]).copy()
    if plot_df.empty:
        return

    groups = sorted(plot_df[group_col].unique())
    n = len(groups)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.1 + 1), 5))

    for i, grp_val in enumerate(groups):
        vals = plot_df.loc[plot_df[group_col] == grp_val, gain_col].dropna()
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   alpha=0.55, s=30, color=COLORS[i % len(COLORS)],
                   edgecolors="white", linewidths=0.3)
        ax.plot([i - 0.3, i + 0.3], [vals.median(), vals.median()],
                color=COLORS[i % len(COLORS)], linewidth=2.5)

    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(n))
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("PCA cosine gain over mean-delta baseline")
    ax.set_title(f"PCA gain distribution by {group_col}\n"
                 "(line = median; dots = individual configs)")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, output_dir, f"pca_gain_by_{group_col}.pdf")


# ── Plot: mean & median of PCA cosine by layer / agg ──────────────────────────

def plot_pca_cosine_by_layer_agg(
        df, output_dir, metric="mean_pca_cosine_reconstructed_eval_delta"):
    if metric not in df.columns:
        return

    group_cols = [c for c in ["layer", "layer_num", "agg_type"] if c in df.columns]
    if not group_cols:
        return

    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return

    summary = (
        plot_df
        .groupby(group_cols, dropna=False)[metric]
        .agg(mean="mean", median="median")
        .reset_index()
    )
    summary["label"] = summary.apply(
        lambda r: " | ".join(str(r[c]) for c in group_cols), axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, len(summary) * 0.38)))

    for ax, stat, color in zip(
            axes,
            ["mean", "median"],
            [PALETTE["blue"], PALETTE["teal"]],
    ):
        s = summary.sort_values(stat, ascending=False)
        bars = ax.barh(s["label"], s[stat], color=color, alpha=0.85)
        for bar, val in zip(bars, s[stat]):
            ax.text(
                bar.get_width() - 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="right", fontsize=7.5,
                color="white", fontweight="bold",
            )
        ax.invert_yaxis()
        ax.set_xlabel("Cosine similarity")
        ax.set_title(f"{stat.capitalize()} of\nmean_pca_cosine_reconstructed_eval_delta\nby layer / agg")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x")

    fig.suptitle("PCA reconstruction cosine by layer & agg type", fontsize=12)
    fig.tight_layout()
    save(fig, output_dir, "mean_median_pca_cosine_reconstructed_eval_delta_by_layer_agg.pdf")


# ── Plot: box plots by layer_agg_boxplot ────────────────────────────────────────
def plot_metric_by_layer_agg_boxplot(
        df,
        output_dir,
        metric,
        xlabel,
        title,
        filename,
        box_color=None,
):
    if metric not in df.columns:
        print(f"[skip] missing metric: {metric}")
        return

    group_cols = [c for c in ["layer", "layer_num", "agg_type"] if c in df.columns]
    if not group_cols:
        return

    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return

    plot_df["label"] = plot_df.apply(
        lambda r: " | ".join(str(r[c]) for c in group_cols),
        axis=1,
    )

    order = (
        plot_df.groupby("label")[metric]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    grouped = [plot_df.loc[plot_df["label"] == lbl, metric].values for lbl in order]

    fig, ax = plt.subplots(figsize=(10, max(5, len(order) * 0.45)))

    bp = ax.boxplot(
        grouped,
        vert=False,
        patch_artist=True,
        tick_labels=order,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#888888"),
        capprops=dict(color="#888888"),
        flierprops=dict(
            marker="o",
            markersize=3,
            alpha=0.5,
            markerfacecolor=PALETTE["coral"],
            markeredgecolor=PALETTE["coral"],
        ),
    )

    if box_color is None:
        box_color = PALETTE["blue"]

    for patch in bp["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_alpha(0.8)

    for i, vals in enumerate(grouped, start=1):
        med = float(np.median(vals))
        ax.text(
            med,
            i - 0.38,
            f"{med:.4f}",
            va="top",
            ha="center",
            fontsize=7.5,
            color="black",
            fontweight="bold",
        )

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x")
    fig.tight_layout()

    save(fig, output_dir, filename)


def plot_metric_by_full_group_boxplot(
        df,
        output_dir,
        metric,
        xlabel,
        title,
        filename,
):
    if metric not in df.columns:
        print(f"[skip] missing metric: {metric}")
        return

    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        print(f"[skip] no rows for metric: {metric}")
        return

    def shorten(val):
        s = str(val or "")
        return s.split("/")[-1] if "/" in s else s

    def make_key(r):
        return (
            str(r.get("model", "") or ""),
            shorten(r.get("train_data_prefix", "")),
            shorten(r.get("eval_data_prefix", "")),
            str(r.get("layer", "") or ""),
            str(r.get("layer_num", "") or ""),
            str(r.get("agg_type", "") or ""),
        )

    plot_df["_key"] = plot_df.apply(make_key, axis=1)
    plot_df["label"] = plot_df["_key"].apply(
        lambda k: f"{k[0]}  tr:{k[1]}  ev:{k[2]}\n{k[3]} | {k[4]} | {k[5]}"
    )

    order = (
        plot_df.groupby("label")[metric]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    label_to_key = dict(zip(plot_df["label"], plot_df["_key"]))
    grouped = [plot_df.loc[plot_df["label"] == lbl, metric].values for lbl in order]

    fig, ax = plt.subplots(figsize=(12, max(6, len(order) * 0.65)))

    bp = ax.boxplot(
        grouped,
        vert=False,
        patch_artist=True,
        tick_labels=[""] * len(order),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#888888"),
        capprops=dict(color="#888888"),
        flierprops=dict(
            marker="o",
            markersize=3,
            alpha=0.5,
            markerfacecolor=PALETTE["coral"],
            markeredgecolor=PALETTE["coral"],
        ),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("#444444")
        patch.set_alpha(0.85)

    for i, vals in enumerate(grouped, start=1):
        med = float(np.median(vals))
        ax.text(
            med,
            i - 0.42,
            f"{med:.4f}",
            va="top",
            ha="center",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="both", colors="black", labelsize=10)
    ax.grid(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fontsize = 10
    gap_pts = 3

    def draw_segments(parts, y_fig_px):
        widths = []
        for txt, _, fw in parts:
            t = ax.annotate(
                txt,
                xy=(0, 0),
                xycoords="axes fraction",
                fontsize=fontsize,
                fontweight=fw,
                annotation_clip=False,
            )
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=renderer)
            widths.append(bb.width)
            t.remove()

        gap_px = gap_pts * fig.dpi / 72
        total_w = sum(widths) + gap_px * (len(parts) - 1)
        x_right = ax.transAxes.transform((0, 0))[0] - 4
        x_cursor = x_right - total_w

        for (txt, color, fw), w in zip(parts, widths):
            x_ax, y_ax = ax.transAxes.inverted().transform((x_cursor, y_fig_px))
            ax.annotate(
                txt,
                xy=(x_ax, y_ax),
                xycoords="axes fraction",
                fontsize=fontsize,
                color=color,
                fontweight=fw,
                ha="left",
                va="center",
                annotation_clip=False,
            )
            x_cursor += w + gap_px

    for i, lbl in enumerate(order, start=1):
        model, train, evl, layer, layer_num, agg = label_to_key[lbl]

        line1_parts = [
            (model, PALETTE["blue"], "bold"),
        ]

        line2_parts = [
            ("  tr:", "#333333", "normal"),
            (train, PALETTE["coral"], "bold"),
            ("  ev:", "#333333", "normal"),
            (evl, "#D4A800", "bold"),
        ]

        line3_parts = [
            (f"{layer} | {layer_num} | {agg}", "#333333", "normal"),
        ]

        y_fig = ax.transData.transform((0, i))[1]
        line_h = fontsize * fig.dpi / 72 * 1.2

        draw_segments(line1_parts, y_fig + line_h * 0.8)
        draw_segments(line2_parts, y_fig)
        draw_segments(line3_parts, y_fig - line_h * 0.8)

    ax.tick_params(axis="y", length=0, labelsize=0)
    fig.subplots_adjust(left=0.35)
    fig.tight_layout()

    save(fig, output_dir, filename)


# ── Original helpers (kept for completeness) ───────────────────────────────────

def plot_bar(df, metric, output_dir, top_k=30, ascending=False):
    if metric not in df.columns:
        return
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return
    plot_df["label"] = plot_df.apply(full_label, axis=1)
    plot_df = plot_df.sort_values(metric, ascending=ascending).head(top_k)

    fig, ax = plt.subplots(figsize=(13, max(5, 0.55 * len(plot_df))))
    ax.barh(plot_df["label"], plot_df[metric],
            color=PALETTE["blue"], alpha=0.82)
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(f"Top {top_k}: {metric.replace('_', ' ')}")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="x")
    fig.tight_layout()
    save(fig, output_dir, f"top_{top_k}_{metric}.pdf")


# ── Plot: per-sample scatter separated by file ────────────────────────────────
def add_saturating_curve(ax, x_vals, y_vals, color="#777777"):
    def saturating_exp(x, y0, ymax, b):
        return ymax - (ymax - y0) * np.exp(-b * x)

    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    ok = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[ok]
    y_vals = y_vals[ok]

    if len(x_vals) < 5:
        return None

    x_min = x_vals.min()
    x_shift = x_vals - x_min

    try:
        fit_result = curve_fit(
            saturating_exp,
            x_shift,
            y_vals,
            p0=[y_vals.min(), y_vals.max(), 5.0],
            bounds=(
                [-np.inf, -np.inf, 0.0],
                [np.inf, np.inf, np.inf],
            ),
            maxfev=10000,
        )

        params = fit_result[0]

        y_pred = saturating_exp(x_shift, *params)

        sat_r = np.corrcoef(y_vals, y_pred)[0, 1]

        sse = np.sum((y_vals - y_pred) ** 2)
        sst = np.sum((y_vals - y_vals.mean()) ** 2)
        sat_r2 = 1.0 - sse / sst if sst > 0 else np.nan

        x_line = np.linspace(x_vals.min(), x_vals.max(), 300)
        y_line = saturating_exp(x_line - x_min, *params)

        ax.plot(
            x_line,
            y_line,
            color=color,
            linewidth=2.0,
            linestyle="-",
            label=f"saturating fit, R²={sat_r2:.3f}",
        )

        return {
            "sat_r": float(sat_r),
            "sat_r2": float(sat_r2),
            "params": [float(x) for x in params],
        }

    except Exception as e:
        print(f"[warn] saturating fit failed: {e}")
        return None


def plot_prompt_subspace_vs_delta_scatter_by_file_single_plot(
        df, output_dir, k,
        x_col="prompt_pca_projection_fraction",
        y_col="pca_explained_l2_fraction_raw",
        suffixes=("all", "last_prompt_token", "mean_prompt"),
):
    """
    For each suffix, save one PDF containing a grid of subplots.

    Row 1: layer_num == 32
    Row 2: layer_num == 64
    Extra rows: any other layer_num values
    """
    import os
    import math

    if x_col not in df.columns or y_col not in df.columns:
        print(f"[skip per-file scatter] missing {x_col} or {y_col}")
        return

    os.makedirs(output_dir, exist_ok=True)

    agg_col = "agg_type" if "agg_type" in df.columns else None
    agg_color = {
        "last_prompt_token": PALETTE["blue"],
        "mean_prompt": PALETTE["teal"],
    }
    suffix_filters = {
        "all": None,
        "last_prompt_token": ("agg_type", "last_prompt_token"),
        "mean_prompt": ("agg_type", "mean_prompt"),
    }

    def _s(val):
        v = str(val or "")
        return v.split("/")[-1] if "/" in v else v

    def config_tag(row):
        return "_".join(filter(None, [
            _s(row.get("model", "").replace("unsloth--", "")),
            _s(row.get("train_data_prefix", "")),
            _s(row.get("eval_data_prefix", "")),
            str(row.get("layer", "") or ""),
            str(row.get("layer_num", "") or ""),
            str(row.get("agg_type", "") or ""),
        ]))

    def get_layer_num(item):
        _, fdata = item
        return str(fdata.iloc[0].get("layer_num", "") or "")

    for suffix in suffixes:
        filt = suffix_filters.get(suffix)

        if filt is not None:
            col, val = filt
            sub_df = df[df[col] == val].copy() if col in df.columns else df.copy()
        else:
            sub_df = df.copy()

        sub_df = sub_df.dropna(subset=[x_col, y_col])
        if sub_df.empty:
            continue

        sub_df["_config_tag"] = sub_df.apply(config_tag, axis=1)
        grouped = list(sub_df.groupby("_config_tag"))

        layer32 = [item for item in grouped if get_layer_num(item) == "32"]
        layer64 = [item for item in grouped if get_layer_num(item) == "64"]

        other_layers = sorted({
            get_layer_num(item)
            for item in grouped
            if get_layer_num(item) not in {"32", "64"}
        })

        ordered_rows = []
        if layer32:
            ordered_rows.append(("layer 32", layer32))
        if layer64:
            ordered_rows.append(("layer 64", layer64))

        for layer_val in other_layers:
            row_items = [item for item in grouped if get_layer_num(item) == layer_val]
            ordered_rows.append((f"{layer_val}", row_items))

        if not ordered_rows:
            continue

        nrows = len(ordered_rows)
        ncols = max(len(row_items) for _, row_items in ordered_rows)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4.5 * nrows),
            squeeze=False,
        )

        for row_idx, (row_label, row_items) in enumerate(ordered_rows):
            for col_idx, (tag, fdata) in enumerate(row_items):
                ax = axes[row_idx][col_idx]

                first = fdata.iloc[0]
                model = _s(first.get("model", ""))
                train = _s(first.get("train_data_prefix", ""))
                evl = _s(first.get("eval_data_prefix", ""))
                layer = str(first.get("layer", "") or "")
                layer_num = str(first.get("layer_num", "") or "")
                agg = str(first.get("agg_type", "") or "")

                if agg_col and agg_col in fdata.columns and fdata[agg_col].nunique() > 1:
                    for grp, gdata in fdata.groupby(agg_col):
                        ax.scatter(
                            gdata[x_col],
                            gdata[y_col],
                            s=12,
                            alpha=0.5,
                            c=agg_color.get(grp, PALETTE["gray"]),
                            edgecolors="none",
                            label=str(grp),
                        )
                    ax.legend(fontsize=6, title=agg_col, title_fontsize=10)
                else:
                    dot_color = agg_color.get(agg, PALETTE["blue"])
                    ax.scatter(
                        fdata[x_col],
                        fdata[y_col],
                        s=12,
                        alpha=0.5,
                        c=dot_color,
                        edgecolors="none",
                    )

                xv = fdata[x_col].values
                yv = fdata[y_col].values

                if len(xv) >= 2:
                    coef = np.polyfit(xv, yv, 1)
                    xl = np.linspace(xv.min(), xv.max(), 200)

                    ax.plot(
                        xl,
                        coef[0] * xl + coef[1],
                        "--",
                        color="#AAAAAA",
                        linewidth=1.0,
                        label="linear trend",
                    )

                    sat_stats = add_saturating_curve(ax, xv, yv)

                    r = np.corrcoef(xv, yv)[0, 1]

                    text = f"linear r = {r:.3f}"
                    if sat_stats is not None:
                        text += f"\nsat r = {sat_stats['sat_r']:.3f}"
                        text += f"\nsat R² = {sat_stats['sat_r2']:.3f}"

                    ax.text(
                        0.03,
                        0.97,
                        text,
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=10,
                        color="#333333",
                    )

                title = (
                    f"{model}\n"
                    f"tr: {train} | ev: {evl}\n"
                    f"layer: {layer} {layer_num}"
                    f"\nagg: {agg}\n"
                    f"n={len(fdata)}"
                )

                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.set_xlabel(x_col.replace("_", " "), fontsize=10)
                ax.set_ylabel(y_col.replace("_", " "), fontsize=10)
                ax.xaxis.label.set_color("black")
                ax.yaxis.label.set_color("black")
                ax.tick_params(axis="both", colors="black", labelsize=10)
                ax.grid(True)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(1.0)

            for col_idx in range(len(row_items), ncols):
                axes[row_idx][col_idx].axis("off")

            axes[row_idx][0].annotate(
                row_label,
                xy=(-0.28, 0.5),
                xycoords="axes fraction",
                rotation=90,
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        x_col_title = x_col.replace("_", " ").replace("pca", "PCA").capitalize()
        y_col_title = y_col.replace("_", " ").replace("pca", "PCA").capitalize()

        fig.suptitle(
            f"{x_col_title} vs {y_col_title} | eval data: {suffix} | pca topk={k}",
            fontsize=14, fontweight="bold"
        )

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = os.path.join(
            output_dir,
            f"{x_col}_vs_{y_col}_{suffix}_subplots.pdf".replace("layer_", "")
        )

        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        print(f"saved {out_path}")


def plot_prompt_subspace_vs_delta_scatter_by_file_single_pdf(
        df, output_dir, k,
        x_col="prompt_pca_projection_fraction",
        y_col="pca_explained_l2_fraction_raw",
        suffixes=("all", "last_prompt_token", "mean_prompt"),
):
    """
    For each suffix, save one multi-page PDF containing one scatter plot per config/file.

    Files are saved as:
        {x_col}_vs_{y_col}_{suffix}_all_configs.pdf
    """
    import os
    from matplotlib.backends.backend_pdf import PdfPages

    if x_col not in df.columns or y_col not in df.columns:
        print(f"[skip per-file scatter] missing {x_col} or {y_col}")
        return

    os.makedirs(output_dir, exist_ok=True)

    agg_col = "agg_type" if "agg_type" in df.columns else None
    agg_color = {
        "last_prompt_token": PALETTE["blue"],
        "mean_prompt": PALETTE["teal"],
    }
    suffix_filters = {
        "all": None,
        "last_prompt_token": ("agg_type", "last_prompt_token"),
        "mean_prompt": ("agg_type", "mean_prompt"),
    }

    def _s(val):
        v = str(val or "")
        return v.split("/")[-1] if "/" in v else v

    def config_tag(row):
        return "_".join(filter(None, [
            _s(row.get("model", "").replace("unsloth--", "")),
            _s(row.get("train_data_prefix", "")),
            _s(row.get("eval_data_prefix", "")),
            str(row.get("layer", "") or ""),
            str(row.get("layer_num", "") or ""),
            str(row.get("agg_type", "") or ""),
        ]))

    for suffix in suffixes:
        filt = suffix_filters.get(suffix)

        if filt is not None:
            col, val = filt
            sub_df = df[df[col] == val].copy() if col in df.columns else df.copy()
        else:
            sub_df = df.copy()

        sub_df = sub_df.dropna(subset=[x_col, y_col])
        if sub_df.empty:
            continue

        sub_df["_config_tag"] = sub_df.apply(config_tag, axis=1)

        pdf_path = os.path.join(
            output_dir,
            f"{x_col}_vs_{y_col}_{suffix}_all_configs.pdf".replace("layer_", "")
        )

        with PdfPages(pdf_path) as pdf:
            for tag, fdata in sub_df.groupby("_config_tag"):
                if fdata.empty:
                    continue

                first = fdata.iloc[0]
                model = _s(first.get("model", ""))
                train = _s(first.get("train_data_prefix", ""))
                evl = _s(first.get("eval_data_prefix", ""))
                layer = str(first.get("layer", "") or "")
                layer_num = str(first.get("layer_num", "") or "")
                agg = str(first.get("agg_type", "") or "")

                fig, ax = plt.subplots(figsize=(7, 6))

                if agg_col and agg_col in fdata.columns and fdata[agg_col].nunique() > 1:
                    for grp, gdata in fdata.groupby(agg_col):
                        ax.scatter(
                            gdata[x_col],
                            gdata[y_col],
                            s=12,
                            alpha=0.5,
                            c=agg_color.get(grp, PALETTE["gray"]),
                            edgecolors="none",
                            label=str(grp),
                        )
                    ax.legend(fontsize=8, title=agg_col, title_fontsize=8)
                else:
                    dot_color = agg_color.get(agg, PALETTE["blue"])
                    ax.scatter(
                        fdata[x_col],
                        fdata[y_col],
                        s=12,
                        alpha=0.5,
                        c=dot_color,
                        edgecolors="none",
                    )

                xv = fdata[x_col].values
                yv = fdata[y_col].values

                if len(xv) >= 2:
                    coef = np.polyfit(xv, yv, 1)
                    xl = np.linspace(xv.min(), xv.max(), 200)

                    ax.plot(
                        xl,
                        coef[0] * xl + coef[1],
                        "--",
                        color="#AAAAAA",
                        linewidth=1.2,
                        label="linear trend",
                    )

                    sat_stats = add_saturating_curve(ax, xv, yv)

                    r = np.corrcoef(xv, yv)[0, 1]
                    ax.legend(fontsize=8)

                    text = f"linear r = {r:.3f}"
                    if sat_stats is not None:
                        text += f"\nsat r = {sat_stats['sat_r']:.3f}"
                        text += f"\nsat R² = {sat_stats['sat_r2']:.3f}"

                    ax.text(
                        0.03,
                        0.97,
                        text,
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=10,
                        color="#333333",
                    )

                title = (
                    f"model: {model}  tr: {train}  ev: {evl}\n"
                    f"layer: {layer}  layer_num: {layer_num}  agg: {agg}\n"
                    f"eval data: {suffix}  n={len(fdata)}\n"
                    f"pca topk={k}"
                )

                ax.set_title(title, fontsize=8)
                ax.set_xlabel(x_col.replace("_", " "))
                ax.set_ylabel(y_col.replace("_", " "))
                ax.grid(True)
                fig.tight_layout()

                pdf.savefig(fig)
                plt.close(fig)

        print(f"saved {pdf_path}")


def plot_prompt_subspace_vs_delta_scatter_by_file(
        df, output_dir, k,
        x_col="prompt_pca_projection_fraction",
        y_col="pca_explained_l2_fraction_raw",
        suffixes=("all", "last_prompt_token", "mean_prompt"),
):
    """
    For each (suffix, file) combination, save one scatter plot image.
    Files are saved as per_file_scatter_{x_col}_vs_{y_col}_{suffix}_{filename}.png
    """
    if x_col not in df.columns or y_col not in df.columns:
        print(f"[skip per-file scatter] missing {x_col} or {y_col}")
        return

    agg_col = "agg_type" if "agg_type" in df.columns else None
    agg_color = {
        "last_prompt_token": PALETTE["blue"],
        "mean_prompt": PALETTE["teal"],
    }
    suffix_filters = {
        "all": None,
        "last_prompt_token": ("agg_type", "last_prompt_token"),
        "mean_prompt": ("agg_type", "mean_prompt"),
    }

    def _s(val):
        v = str(val or "")
        return v.split("/")[-1] if "/" in v else v

    def config_tag(row):
        return "_".join(filter(None, [
            _s(row.get("model", "").replace("unsloth--", "")),
            _s(row.get("train_data_prefix", "")),
            _s(row.get("eval_data_prefix", "")),
            str(row.get("layer", "") or ""),
            str(row.get("layer_num", "") or ""),
            str(row.get("agg_type", "") or ""),
        ]))

    for suffix in suffixes:
        filt = suffix_filters.get(suffix)
        if filt is not None:
            col, val = filt
            sub_df = df[df[col] == val].copy() if col in df.columns else df.copy()
        else:
            sub_df = df.copy()

        sub_df = sub_df.dropna(subset=[x_col, y_col])
        if sub_df.empty:
            continue

        sub_df["_config_tag"] = sub_df.apply(config_tag, axis=1)

        for tag, fdata in sub_df.groupby("_config_tag"):
            if fdata.empty:
                continue

            first = fdata.iloc[0]
            model = _s(first.get("model", ""))
            train = _s(first.get("train_data_prefix", ""))
            evl = _s(first.get("eval_data_prefix", ""))
            layer = str(first.get("layer", "") or "")
            layer_num = str(first.get("layer_num", "") or "")
            agg = str(first.get("agg_type", "") or "")

            fig, ax = plt.subplots(figsize=(7, 6))

            # Color by agg_type if multiple present (only relevant for suffix=all)
            if agg_col and agg_col in fdata.columns and fdata[agg_col].nunique() > 1:
                for grp, gdata in fdata.groupby(agg_col):
                    ax.scatter(
                        gdata[x_col], gdata[y_col],
                        s=12, alpha=0.5,
                        c=agg_color.get(grp, PALETTE["gray"]),
                        edgecolors="none", label=str(grp),
                    )
                ax.legend(fontsize=8, title=agg_col, title_fontsize=8)
            else:
                dot_color = agg_color.get(agg, PALETTE["blue"])
                ax.scatter(fdata[x_col], fdata[y_col],
                           s=12, alpha=0.5, c=dot_color, edgecolors="none")

            # Trend line + Pearson r
            xv = fdata[x_col].values
            yv = fdata[y_col].values
            if len(xv) >= 2:
                # fiting curves
                coef = np.polyfit(xv, yv, 1)
                xl = np.linspace(xv.min(), xv.max(), 200)

                ax.plot(
                    xl,
                    coef[0] * xl + coef[1],
                    "--",
                    color="#AAAAAA",
                    linewidth=1.2,
                    label="linear trend",
                )

                sat_stats = add_saturating_curve(ax, xv, yv)

                r = np.corrcoef(xv, yv)[0, 1]
                ax.legend(fontsize=8)
                text = f"linear r = {r:.3f}"
                if sat_stats is not None:
                    text += f"\nsat r = {sat_stats['sat_r']:.3f}"
                    text += f"\nsat R² = {sat_stats['sat_r2']:.3f}"

                ax.text(
                    0.03,
                    0.97,
                    text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=10,
                    color="#333333",
                )

            title = (f"model: {model}  tr: {train}  ev: {evl}\n"
                     f"layer: {layer}  layer_num: {layer_num}  agg: {agg}\n"
                     f"eval data: {suffix}  n={len(fdata)}\n"
                     f"pca topk={k}"
                     )
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(x_col.replace("_", " "))
            ax.set_ylabel(y_col.replace("_", " "))
            ax.grid(True)
            fig.tight_layout()

            save(fig, output_dir,
                 f"{x_col}_vs_{y_col}_{suffix}_{tag}.pdf".replace("layer_", ""))


def plot_metric_scatter(
        df,
        output_dir,
        x_col,
        y_col,
        filename,
        xlabel=None,
        ylabel=None,
        title=None,
        color_by="agg_type",
        point_size=40,
        alpha=0.65,
):
    if x_col not in df.columns or y_col not in df.columns:
        print(f"[skip] missing {x_col} or {y_col}")
        return

    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        print(f"[skip] no rows for {x_col} vs {y_col}")
        return

    c_col = color_by if color_by in plot_df.columns else None

    color_map = {
        "last_prompt_token": PALETTE["darker-blue"],
        "mean_prompt": PALETTE["darker-green"],
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    if c_col:
        for grp, sub in plot_df.groupby(c_col):
            color = color_map.get(grp, PALETTE["gray"])

            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=point_size,
                alpha=alpha,
                c=color,
                edgecolors="white",
                linewidths=0.4,
                label=str(grp),
            )

            x_vals = sub[x_col].values
            y_vals = sub[y_col].values

            if len(sub) >= 2:
                coef = np.polyfit(x_vals, y_vals, deg=1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
                y_line = coef[0] * x_line + coef[1]

                ax.plot(
                    x_line,
                    y_line,
                    "--",
                    color=color,
                    linewidth=1.2,
                    label=f"{grp} linear trend",
                )

                sat_stats = add_saturating_curve(
                    ax,
                    x_vals,
                    y_vals,
                    color=color,
                )

                r = np.corrcoef(x_vals, y_vals)[0, 1]

                text = f"{grp}\nlinear r = {r:.3f}"
                if sat_stats is not None:
                    text += f"\nsat r = {sat_stats['sat_r']:.3f}"
                    text += f"\nsat R² = {sat_stats['sat_r2']:.3f}"

                ax.text(
                    0.03,
                    0.97 - 0.16 * list(plot_df.groupby(c_col).groups).index(grp),
                    text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=10,
                    color=color,
                )

    else:
        ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            s=point_size,
            alpha=alpha,
            c=PALETTE["gray"],
            edgecolors="white",
            linewidths=0.4,
        )

        x_vals = plot_df[x_col].values
        y_vals = plot_df[y_col].values

        if len(plot_df) >= 2:
            coef = np.polyfit(x_vals, y_vals, deg=1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
            y_line = coef[0] * x_line + coef[1]

            ax.plot(
                x_line,
                y_line,
                "--",
                color="#AAAAAA",
                linewidth=1.2,
                label="linear trend",
            )

            sat_stats = add_saturating_curve(ax, x_vals, y_vals)

            r = np.corrcoef(x_vals, y_vals)[0, 1]

            text = f"linear r = {r:.3f}"
            if sat_stats is not None:
                text += f"\nsat r = {sat_stats['sat_r']:.3f}"
                text += f"\nsat R² = {sat_stats['sat_r2']:.3f}"

            ax.text(
                0.03,
                0.97,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
            )

    ax.legend(
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    ax.set_xlabel(xlabel or x_col.replace("_", " "), color="black", fontsize=12)
    ax.set_ylabel(ylabel or y_col.replace("_", " "), color="black", fontsize=12)
    ax.set_title(title or f"{y_col} vs {x_col}", fontweight="bold", fontsize=14)
    ax.grid(True)

    # Add subtle border around plot
    # Make axes, ticks, and axis labels black
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()

    save(fig, output_dir, filename)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
        topk,
        input_dir="./eval_deltas_vs_train_delta_pca",
        output_dir_prefix="./deltas/delta_comparison_analysis",
):
    output_dir = ".".join([output_dir_prefix, f"k{topk}"])
    os.makedirs(output_dir, exist_ok=True)

    df = load_delta_results(input_dir, topk=topk)
    df = add_metrics(df)

    csv_path = os.path.join(output_dir, "all_delta_comparison_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    sample_df = load_per_sample_delta_results(input_dir, topk=topk)
    eps = 1e-12
    sample_df["eval_delta_norm_backed_out"] = (
            sample_df["mean_delta_scalar_projection"].abs()
            / sample_df["mean_delta_cosine"].abs().clip(lower=eps)
    )

    sample_csv_path = os.path.join(output_dir, "all_per_sample_delta_metrics.csv")
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"[saved] {sample_csv_path}")

    # ── New richer comparison plots ──────────────────────────────────────────

    # 1. All-in-one heatmap (all configs)
    plot_heatmap(df, output_dir)

    # 2. Bubble scatter (3 dimensions at once)
    plot_bubble_scatter(df, output_dir)

    # 3 scatter plots
    plot_metric_scatter(
        df,
        output_dir,
        x_col="mean_pca_explained_l2_fraction_raw",
        y_col="mean_pca_cosine_reconstructed_eval_delta",
        filename="scatter_mean_pca_explained_l2_fraction_vs_delta_recon_cosine.pdf",
        xlabel="PCA explained l2 fraction",
        ylabel="PCA reconstruction cosine",
        title=f"PCA explained l2 fraction vs PCA reconstruction cosine\npca topk = {topk}",
        point_size=12,
        alpha=0.45,
    )

    plot_metric_scatter(
        df,
        output_dir,
        x_col="mean_cosine_eval_delta_train_mean_delta",
        y_col="mean_pca_cosine_reconstructed_eval_delta",
        filename="scatter_delta_recon_cosine_vs_delta_cosine.pdf",
        xlabel="PCA explained l2 fraction",
        ylabel="delta cosine",
        title=f"PCA explained l2 fraction vs PCA reconstruction cosine\npca topk = {topk}",
        point_size=12,
        alpha=0.45,
    )

    plot_metric_scatter(
        sample_df,
        output_dir,
        x_col="pca_explained_l2_fraction_raw",
        y_col="pca_cosine_reconstructed_eval_delta",
        filename="scatter_mean_pca_explained_l2_fraction_vs_delta_recon_cosine_per_sample.pdf",
        xlabel="PCA explained l2 fraction",
        ylabel="PCA reconstruction cosine",
        title=f"PCA explained l2 fraction vs PCA reconstruction cosine\nper eval\npca topk = {topk}",
        point_size=12,
        alpha=0.45,
    )

    plot_metric_scatter(
        sample_df,
        output_dir,
        x_col="prompt_pca_projection_fraction",
        y_col="pca_cosine_reconstructed_eval_delta",
        filename="scatter_prompt_pca_projection_fraction_vs_pca_cosine_reconstructed_eval_delta.pdf",
        xlabel="Prompt overlap",
        ylabel="Delta reconstruction cosine",
        title=f"Prompt overlap vs PCA reconstruction cosine\npca topk = {topk}",
        point_size=12,
        alpha=0.45,
    )

    plot_metric_scatter(
        sample_df,
        output_dir,
        x_col="prompt_pca_projection_fraction",
        y_col="eval_delta_norm_backed_out",
        filename="scatter_prompt_pca_projection_fraction_vs_eval_delta_norm_backed_out.pdf",
        xlabel="Prompt PCA projection fraction",
        ylabel="Backed-out eval delta norm",
        title=f"Prompt overlap vs eval delta norm\npca topk = {topk}",
        point_size=12,
        alpha=0.45,
    )

    plot_metric_scatter(
        sample_df,
        output_dir,
        x_col="prompt_pca_projection_fraction",
        y_col="mean_delta_scalar_projection",
        filename="scatter_prompt_pca_projection_fraction_vs_mean_delta_scalar_projection.pdf",
        xlabel="Prompt PCA projection fraction",
        ylabel="Mean delta scalar projection",
        title=f"Prompt overlap vs Mean delta scalar projection\npca topk = {topk}",
        point_size=12,
        alpha=0.45,
    )

    plot_metric_scatter(
        sample_df,
        output_dir,
        x_col="prompt_pca_projection_fraction",
        y_col="mean_delta_cosine",
        filename="scatter_prompt_pca_projection_fraction_vs_mean_delta_cosine.pdf",
        xlabel="Prompt overlap",
        ylabel="Delta cosine similarity",
        title=f"Prompt overlap vs delta cosine similarity\npca topk = {topk}",
        point_size=12,
        alpha=0.45,
    )

    plot_metric_scatter(
        sample_df,
        output_dir,
        x_col="prompt_pca_projection_fraction",
        y_col="mean_delta_projection_fraction",
        filename="scatter_prompt_pca_projection_fraction_vs_mean_delta_projection_fraction.pdf",
        xlabel="Prompt overlap",
        ylabel="Delta projection",
        title=f"Prompt overlap vs delta projection\npca topk = {topk}",
        point_size=12,
        alpha=0.45,
    )

    # 4. PCA gain strip plot grouped by activation type / agg type
    plot_gain_by_group(df, output_dir)

    # 5. Mean & median PCA cosine by layer / agg
    plot_pca_cosine_by_layer_agg(df, output_dir)

    # 6. Boxplot version full
    plot_metric_by_full_group_boxplot(
        sample_df,
        output_dir,
        metric="pca_cosine_reconstructed_eval_delta",
        xlabel="Reconstruct vs. true eval delta cosine similarity",
        title="PCA reconstruction cosine - full group boxplot\n"
              "(model × train × eval × layer × layer_num × agg_type; sorted by median)\n"
              f"pca topk = {topk}",
        filename="boxplot_pca_cosine_reconstructed_eval_delta_by_full_group.pdf",
    )

    plot_metric_by_full_group_boxplot(
        sample_df,
        output_dir,
        metric="pca_explained_l2_fraction_raw",
        xlabel="PCA explained L2 fraction",
        title="PCA explained L2 fraction - full group boxplot\n"
              "(model × train × eval × layer × layer_num × agg_type; sorted by median)\n"
              f"pca topk = {topk}",
        filename="boxplot_pca_explained_l2_fraction_by_full_group.pdf",
    )

    plot_metric_by_full_group_boxplot(
        sample_df,
        output_dir,
        metric="mean_delta_cosine",
        xlabel="Delta cosine similarity",
        title="Delta cosine similarity - full group boxplot\n"
              "(model × train × eval × layer × layer_num × agg_type; sorted by median)\n",
        filename="boxplot_mean_delta_cosine_by_full_group.pdf",
    )
    # pca_mahalanobis_distance
    plot_metric_by_full_group_boxplot(
        sample_df,
        output_dir,
        metric="pca_mahalanobis_distance",
        xlabel="Mahalanobi Distance",
        title="Mahalanobi Distance - full group boxplot\n"
              "(model × train × eval × layer × layer_num × agg_type; sorted by median)\n",
        filename="boxplot_pca_mahalanobis_distance_by_full_group.pdf",
    )

    # agg
    plot_metric_by_layer_agg_boxplot(
        df,
        output_dir,
        metric="mean_pca_cosine_reconstructed_eval_delta",
        xlabel="Cosine similarity",
        title="PCA reconstruction cosine - by layer & agg type — boxplot\n"
              "(sorted by median; label: layer | layer_num | agg_type)\n"
              f"pca topk = {topk}",
        filename="boxplot_pca_cosine_reconstructed_eval_delta_by_layer_agg.pdf",
        box_color=PALETTE["blue"],
    )

    plot_metric_by_layer_agg_boxplot(
        df,
        output_dir,
        metric="mean_pca_explained_l2_fraction_raw",
        xlabel="PCA explained L2 fraction",
        title="PCA explained L2 fraction - by layer & agg type — boxplot\n"
              "(sorted by median; label: layer | layer_num | agg_type)\n"
              f"pca topk = {topk}",
        filename="boxplot_pca_explained_l2_fraction_by_layer_agg.pdf",
        box_color=PALETTE["purple"],
    )

    # 10. Per-sample scatter split by file
    for x_col, y_col, suffix_list in [
        (
                "prompt_pca_projection_fraction",
                "pca_explained_l2_fraction_raw",
                ["last_prompt_token", "mean_prompt"],
        ),
        (
                "prompt_pca_projection_fraction",
                "pca_cosine_reconstructed_eval_delta",
                ["last_prompt_token", "mean_prompt"],
        ),
        (
                "prompt_pca_projection_fraction",
                "pca_cosine_reconstructed_eval_delta",
                ["last_prompt_token", "mean_prompt"],
        ),
        (
                "prompt_pca_projection_fraction",
                "mean_delta_cosine",
                ["last_prompt_token", "mean_prompt"],
        ),
        (
                "prompt_pca_projection_fraction",
                "mean_delta_scalar_projection",
                ["last_prompt_token", "mean_prompt"],
        ),
        (
                "random_projection_fraction",
                "pca_explained_l2_fraction_raw",
                ["last_prompt_token", "mean_prompt"],
        ),
        (
                "random_projection_fraction",
                "pca_cosine_reconstructed_eval_delta",
                ["last_prompt_token", "mean_prompt"],
        ),
        (
                "random_projection_fraction",
                "mean_delta_cosine",
                ["last_prompt_token", "mean_prompt"],
        ),
        # pca_mahalanobis_distance
        (
                "prompt_pca_mahalanobis_distance",
                "pca_mahalanobis_distance",
                ["last_prompt_token", "mean_prompt"],
        ),
    ]:
        plot_prompt_subspace_vs_delta_scatter_by_file_single_plot(
            sample_df, output_dir, topk,
            x_col=x_col, y_col=y_col, suffixes=suffix_list,
        )

    # for x_col, y_col, suffix_list in [
    #     (
    #             "prompt_pca_projection_fraction",
    #             "pca_cosine_reconstructed_eval_delta",
    #             ["last_prompt_token", "mean_prompt"],
    #     ),
    # ]:
    #     plot_prompt_subspace_vs_delta_scatter_by_file(
    #         sample_df, output_dir, topk,
    #         x_col=x_col, y_col=y_col, suffixes=suffix_list,
    #     )

    # ── Console summary ──────────────────────────────────────────────────────

    cols = [
        "mean_cosine_eval_delta_train_mean_delta",
        "mean_pca_explained_l2_fraction_raw",
        "mean_pca_cosine_reconstructed_eval_delta",
        "pca_vs_mean_cosine_gain",
        "activation_type", "layer", "layer_num", "agg_type",
    ]
    available = [c for c in cols if c in df.columns]

    print("\nBest by mean-delta cosine:")
    print(df.sort_values("mean_cosine_eval_delta_train_mean_delta",
                         ascending=False)[available].head(10).to_string(index=False))

    print("\nBest by PCA reconstruction cosine:")
    print(df.sort_values("mean_pca_cosine_reconstructed_eval_delta",
                         ascending=False)[available].head(10).to_string(index=False))

    print("\nBest by PCA L2 explained fraction:")
    if "mean_pca_explained_l2_fraction_raw" in df.columns:
        print(df.sort_values("mean_pca_explained_l2_fraction_raw",
                             ascending=False)[available].head(10).to_string(index=False))


if __name__ == "__main__":
    # import fire; fire.Fire(main)
    # all_k = [1, 2, 4, 8, 16, 32, 64, 128]
    all_k = [0.8, 0.9, 0.95]
    # all_k = [8]
    for topk in all_k:
        print(topk)
        if topk < 1:
            input_dir = "./eval_deltas_vs_train_delta_metrics_95"
            output_dir_prefix = "./deltas/delta_comparison_analysis_95"
        else:
            input_dir = "./eval_deltas_vs_train_delta_pca"
            output_dir_prefix = "./deltas/delta_comparison_analysis"
        main(topk=topk, input_dir=input_dir, output_dir_prefix=output_dir_prefix)
