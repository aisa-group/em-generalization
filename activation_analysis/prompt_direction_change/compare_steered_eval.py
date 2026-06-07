#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def flatten_json(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_json(v, key))
        else:
            out[key] = v
    return out


def load_results(root: Path) -> pd.DataFrame:
    rows = []

    files = sorted(root.rglob("*.steered_eval.comparison.json"))

    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        row = flatten_json(data)

        row["file_path"] = str(path)
        row["file_name"] = path.name

        row["parent_folder"] = path.parent.name
        row["layer_folder"] = path.parent.name
        row["layer_type_folder"] = path.parent.parent.name if path.parent.parent else None
        row["activation_type_folder"] = (
            path.parent.parent.parent.name if path.parent.parent.parent else None
        )
        try:
            row["source_dir"] = path.relative_to(root).parts[0]
        except Exception:
            row["source_dir"] = path.parts[0]

        rows.append(row)

    return pd.DataFrame(rows)


def add_improvement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    lower_is_better = [
        "mse",
        "rmse",
        "mae",
        "mean_l2_error",
        "median_l2_error",
        "kl_true_to_compare",
        "kl_compare_to_true",
        "symmetric_kl",
    ]

    for metric in lower_is_better:
        u = f"comparison.unsteered_true.unsteered_true_{metric}"
        s = f"comparison.steered_true.steered_true_{metric}"

        if u in df.columns and s in df.columns:
            df[f"improvement.{metric}.absolute"] = df[u] - df[s]
            df[f"improvement.{metric}.percent"] = 100 * (df[u] - df[s]) / df[u]

    higher_is_better = [
        "mean_cosine",
        "median_cosine",
    ]

    for metric in higher_is_better:
        u = f"comparison.unsteered_true.unsteered_true_{metric}"
        s = f"comparison.steered_true.steered_true_{metric}"

        if u in df.columns and s in df.columns:
            df[f"improvement.{metric}.absolute"] = df[s] - df[u]
            df[f"improvement.{metric}.percent"] = 100 * (df[s] - df[u]) / df[u]

    return df


def save_grouped_summary(df: pd.DataFrame, out_path: Path, top_k):
    group_cols = [
        "config.model",
        "config.train_data_prefix",
        "config.eval_data_prefix",
        "config.activation_type",
        "config.layer",
        "config.layer_num",
        "config.agg_type",
        "config.shift_key",
        "config.alpha",
        "config.mode",
    ]

    group_cols = [c for c in group_cols if c in df.columns]

    if not group_cols or "improvement.symmetric_kl.percent" not in df.columns:
        return

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("file_path", "count"),
            mean_symmetric_kl_improvement=("improvement.symmetric_kl.percent", "mean"),
            median_symmetric_kl_improvement=("improvement.symmetric_kl.percent", "median"),
            mean_rmse_improvement=("improvement.rmse.percent", "mean"),
            mean_mae_improvement=("improvement.mae.percent", "mean"),
            mean_cosine_gain=("improvement.mean_cosine.absolute", "mean"),
        )
        .reset_index()
        .sort_values("mean_symmetric_kl_improvement", ascending=False)
    )

    grouped_path = out_path.with_name(out_path.stem + "_grouped.csv")
    grouped.to_csv(grouped_path, index=False)
    print(f"Saved grouped summary: {grouped_path}")

    print("\nTop grouped configs:")
    if top_k is None:
        print(grouped.to_string(index=False))
    else:
        print(grouped.head(top_k).to_string(index=False))


def make_plots(
        df: pd.DataFrame,
        out_dir: Path,
        x_col="comparison.unsteered_true.unsteered_true_mean_cosine",
        y_col="comparison.steered_true.steered_true_mean_cosine",
        save_name="mean_cosine",
        title_name="Mean Cosine"
):
    out_dir.mkdir(parents=True, exist_ok=True)

    def scatter_mean_cosine_with_labels():
        from matplotlib.lines import Line2D

        layer_col = "config.layer_num"

        if not {x_col, y_col, layer_col}.issubset(df.columns):
            return

        layers = sorted(df[layer_col].dropna().unique())

        fig, axes = plt.subplots(
            1,
            len(layers),
            figsize=(8 * len(layers), 7),
            sharex=True,
            sharey=True,
        )

        if len(layers) == 1:
            axes = [axes]

        lo = min(df[x_col].min(), df[y_col].min())
        hi = max(df[x_col].max(), df[y_col].max())

        legend_items = [
            Line2D([0], [0], marker="o", color="w", label="mean_prompt_tokens",
                   markerfacecolor="green", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="last_prompt_token",
                   markerfacecolor="blue", markersize=8),
            Line2D([0], [0], marker="x", color="black", label="alpha=0.5 raw",
                   markersize=8),
            Line2D([0], [0], marker="o", color="black", label="alpha=1.0 raw",
                   markersize=8),
        ]

        for ax, layer in zip(axes, layers):
            layer_df = df[df[layer_col] == layer]

            for marker in ["x", "o", "."]:
                subset = layer_df[layer_df.apply(get_marker, axis=1) == marker]

                if subset.empty:
                    continue

                colors = subset.apply(get_color, axis=1)

                ax.scatter(
                    subset[x_col],
                    subset[y_col],
                    c=colors,
                    marker=marker,
                    alpha=0.75,
                )

            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", alpha=0.7)

            if "cosine" in x_col:
                label_thres = layer_df[layer_df[y_col] < layer_df[x_col]]
            else:
                label_thres = layer_df[layer_df[y_col] > layer_df[x_col]]

            for _, row in label_thres.iterrows():
                label = row.get("source_dir", "unknown")

                ax.annotate(
                    label,
                    xy=(row[x_col], row[y_col]),
                    xytext=(5, -5),
                    textcoords="offset points",
                    fontsize=10,
                    alpha=0.85,
                )

            ax.set_title(str(layer), fontsize=12)
            ax.set_xlabel("Unsteered mean cosine", fontsize=12)
            if ax is axes[0]:
                ax.set_ylabel("Steered mean cosine", fontsize=12)
            else:
                ax.set_ylabel("")
            ax.legend(handles=legend_items, loc="best", fontsize=12, frameon=True)

        fig.suptitle(
            f"{title_name}: Unsteered vs Steered by Layer",
            fontsize=16,
            y=0.98,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        save(f"{save_name}_unsteered_vs_steered_by_layer.png")

    def save(name):
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved plot: {path}")

    def get_color(row):
        agg = str(row.get("config.agg_type", "")).lower()
        if agg == "mean_prompt":
            return "green"
        if agg == "last_prompt_token":
            return "blue"
        if agg == "last_prompt_tokens":
            return "blue"
        return "gray"

    def get_marker(row):
        alpha = row.get("config.alpha", None)
        mode = str(row.get("config.mode", "")).lower()

        try:
            alpha = float(alpha)
        except Exception:
            alpha = None

        if mode == "raw" and alpha == 0.5:
            return "x"
        if mode == "raw" and alpha == 1.0:
            return "o"
        return "."

    def scatter_before_after(unsteered_col, steered_col, title, xlabel, ylabel, filename):
        if not {unsteered_col, steered_col}.issubset(df.columns):
            return

        plt.figure(figsize=(8, 7))

        for marker in ["x", "o", "."]:
            subset = df[df.apply(get_marker, axis=1) == marker]

            if subset.empty:
                continue

            colors = subset.apply(get_color, axis=1)

            plt.scatter(
                subset[unsteered_col],
                subset[steered_col],
                c=colors,
                marker=marker,
                alpha=0.75,
                label={
                    "x": "alpha=0.5 raw",
                    "o": "alpha=1.0 raw",
                    ".": "other",
                }[marker],
            )

        lo = min(df[unsteered_col].min(), df[steered_col].min())
        hi = max(df[unsteered_col].max(), df[steered_col].max())

        plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", alpha=0.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Custom color legend
        from matplotlib.lines import Line2D

        legend_items = [
            Line2D([0], [0], marker="o", color="w", label="mean_prompt_tokens", markerfacecolor="green", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="last_prompt_token", markerfacecolor="blue", markersize=8),
            Line2D([0], [0], marker="x", color="black", label="alpha=0.5 raw", markersize=8),
            Line2D([0], [0], marker="o", color="black", label="alpha=1.0 raw", markersize=8),
        ]

        plt.legend(handles=legend_items)
        save(filename)

    # Before/after scatter plots
    scatter_mean_cosine_with_labels()

    scatter_before_after(
        "comparison.unsteered_true.unsteered_true_rmse",
        "comparison.steered_true.steered_true_rmse",
        "RMSE: Unsteered vs Steered (lower is better)",
        "Unsteered RMSE",
        "Steered RMSE",
        "rmse_unsteered_vs_steered.png",
    )

    scatter_before_after(
        "comparison.unsteered_true.unsteered_true_symmetric_kl",
        "comparison.steered_true.steered_true_symmetric_kl",
        "Symmetric KL: Unsteered vs Steered (lower is better)",
        "Unsteered symmetric KL",
        "Steered symmetric KL",
        "symmetric_kl_unsteered_vs_steered.png",
    )


def plot_steered_improvements(df: pd.DataFrame, out_dir: Path):
    """
    Plot percent improvement of steered over unsteered as three subplots in
    one figure. All metrics are expressed as "% improvement over unsteered",
    so for every panel higher (more positive) is better.

    Metrics:
    - RMSE              (raw: lower is better -> improvement positive = good)
    - Symmetric KL      (raw: lower is better -> improvement positive = good)
    - Mean cosine sim   (raw: higher is better -> improvement positive = good)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("rmse", "RMSE", "improvement.rmse.percent"),
        # ("symmetric_kl", "Symmetric KL", "improvement.symmetric_kl.percent"),
        ("mean_cosine", "Mean cosine", "improvement.mean_cosine.percent"),
    ]

    # Keep only metrics whose column actually exists.
    metrics = [m for m in metrics if m[2] in df.columns]
    if not metrics:
        print("No improvement columns found; nothing to plot.")
        return

    label_cols = [
        "config.model",  # model first
        "config.train_data_prefix",
        "config.eval_data_prefix",
        "config.dataset",
        "config.agg_type",
        "config.alpha",
        "config.mode",
        "config.layer_num",
    ]
    label_cols = [c for c in label_cols if c in df.columns]

    plot_df = df.copy()
    if label_cols:
        plot_df["plot_label"] = plot_df[label_cols].astype(str).agg(" | ".join, axis=1)
    else:
        plot_df["plot_label"] = plot_df.index.astype(str)

    # Use a single, consistent config ordering across all subplots so a given
    # row corresponds to the same config in every panel. Order by the first
    # available metric (descending), falling back to rows that have any data.
    sort_col = metrics[0][2]
    ordered = plot_df.dropna(subset=[c for _, _, c in metrics], how="all")
    ordered = ordered.sort_values(sort_col, ascending=True)  # ascending: best ends up on top after barh
    labels = ordered["plot_label"].tolist()
    n_rows = len(labels)

    if n_rows == 0:
        print("No rows with valid improvement values; nothing to plot.")
        return

    y = range(n_rows)
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(6 * len(metrics), max(4, 0.4 * n_rows)),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_name, title, col) in zip(axes, metrics):
        values = ordered[col].to_numpy(dtype=float)
        colors = ["#2ca02c" if (v is not None and v >= 0) else "#d62728" for v in values]

        ax.barh(list(y), values, color=colors)
        ax.axvline(0, linestyle="--", color="black", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Improvement over\nunsteered (%)")
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        # Annotate each bar with its value.
        for yi, v in zip(y, values):
            if pd.isna(v):
                continue
            ax.text(
                v,
                yi,
                f" {v:+.1f}%",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )

    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels)
    axes[0].set_ylabel("Config")

    fig.suptitle("Steered vs. unsteered improvement", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = out_dir / "steered_improvement_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_path}")


def main():
    # =========================
    # Edit these values directly
    # =========================
    ROOT = "steered_eval/"
    OUT = "steered_eval/results.csv"

    # Use None to graph/print all configs.
    # Use an int like 20 or 50 to limit.
    TOP_K = None

    root = Path(ROOT)
    out_path = Path(OUT)

    df = load_results(root)

    if df.empty:
        print("No *.steered_eval.comparison.json files found.")
        return

    df = add_improvement_metrics(df)

    df.to_csv(out_path, index=False)
    print(f"Loaded {len(df)} files")
    print(f"Saved full summary: {out_path}")

    sort_col = "improvement.symmetric_kl.percent"

    if sort_col in df.columns:
        print("\nTop individual configs:")

        view_cols = [
            "file_path",
            "config.model",
            "config.train_data_prefix",
            "config.eval_data_prefix",
            "config.activation_type",
            "config.layer",
            "config.layer_num",
            "config.agg_type",
            "config.shift_key",
            "config.alpha",
            "config.mode",
            "comparison.unsteered_true.unsteered_true_symmetric_kl",
            "comparison.steered_true.steered_true_symmetric_kl",
            "improvement.symmetric_kl.percent",
            "comparison.unsteered_true.unsteered_true_mean_cosine",
            "comparison.steered_true.steered_true_mean_cosine",
            "improvement.mean_cosine.absolute",
        ]

        view_cols = [c for c in view_cols if c in df.columns]
        ranked = df.sort_values(sort_col, ascending=False)

        if TOP_K is None:
            print(ranked[view_cols].to_string(index=False))
        else:
            print(ranked.head(TOP_K)[view_cols].to_string(index=False))

    save_grouped_summary(df, out_path, TOP_K)

    plot_dir = Path("steered_eval_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for x, y, save_name, title_name in [
        (
                "comparison.unsteered_true.unsteered_true_mean_cosine",
                "comparison.steered_true.steered_true_mean_cosine",
                "mean_cosine",
                "Mean Cosine"
        ),
        (
                "comparison.unsteered_true.unsteered_true_symmetric_kl",
                "comparison.steered_true.steered_true_symmetric_kl",
                "symmetric_kl",
                "Symmetric KL (lower is better)"
        ),
        (
                "comparison.unsteered_true.unsteered_true_rmse",
                "comparison.steered_true.steered_true_rmse",
                "rmse",
                "RMSE (lower is better)"
        ),

    ]:
        make_plots(
            df, plot_dir,
            x_col=x,
            y_col=y,
            save_name=save_name,
            title_name=title_name,
        )

    plot_steered_improvements(df, plot_dir)
    plot_steered_raw_values(df, plot_dir)


if __name__ == "__main__":
    main()
