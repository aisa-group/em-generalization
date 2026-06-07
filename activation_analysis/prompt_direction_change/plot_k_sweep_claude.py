"""
Compare delta comparison metrics across k{pc_num} PCA components.

Expects folders named: delta_comparison_analysis.k1, delta_comparison_analysis.k2, ...
Each containing: all_delta_comparison_metrics.csv
"""

import glob
import re
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


def load_all_k_results(base_dir: str = ".") -> pd.DataFrame:
    """Load all_delta_comparison_metrics.csv from each delta_comparison_analysis.k* folder."""
    pattern = str(Path(base_dir) / "delta_comparison_analysis.k*" / "all_delta_comparison_metrics.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    dfs = []
    for f in files:
        match = re.search(r"delta_comparison_analysis\.k(\d+)", f)
        k = int(match.group(1)) if match else None

        df = pd.read_csv(f)
        df["k"] = k
        dfs.append(df)
        print(f"  Loaded k={k}: {len(df)} rows from {f}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows across all k: {len(combined)}")
    return combined


def summarize_by_k(df: pd.DataFrame, metrics: Optional[list[str]] = None) -> pd.DataFrame:
    """Mean of each metric grouped by k."""
    if metrics is None:
        metrics = [
            "mean_pca_projection_fraction",
            "mean_pca_residual_fraction",
            "mean_pca_explained_l2_fraction_raw",
            "mean_pca_cosine_reconstructed_eval_delta",
            "mean_cosine_eval_delta_train_mean_delta",
            "pca_vs_mean_cosine_gain",
            "pca_projection_minus_residual_fraction",
        ]
        metrics = [m for m in metrics if m in df.columns]

    return df.groupby("k")[metrics].mean().round(4)


def compare_by_k_and_group(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str = "mean_cosine_eval_delta_train_mean_delta",
) -> pd.DataFrame:
    """Pivot: rows = group_cols, columns = k values, values = metric."""
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")

    pivot = df.pivot_table(index=group_cols, columns="k", values=metric, aggfunc="mean")
    pivot.columns = [f"k={c}" for c in pivot.columns]

    # Add delta columns relative to k=1
    k1_col = "k=1"
    if k1_col in pivot.columns:
        for col in pivot.columns:
            if col != k1_col:
                pivot[f"Δ({col}-k=1)"] = pivot[col] - pivot[k1_col]

    return pivot.round(4)


PLOT_METRICS = [
    ("mean_cosine_eval_delta_train_mean_delta", "Mean Cosine Similarity\n(eval delta vs train mean delta)"),
    ("mean_pca_projection_fraction",            "Mean PCA Projection Fraction"),
    ("mean_pca_residual_fraction",              "Mean PCA Residual Fraction"),
    ("mean_pca_explained_l2_fraction_raw",      "Mean PCA Explained L2 Fraction (raw)"),
    ("pca_vs_mean_cosine_gain",                 "PCA vs Mean Cosine Gain"),
    ("mean_pca_cosine_reconstructed_eval_delta", "Actual vs PCA reconstructed Mean Cosine")
    # ("pca_projection_minus_residual_fraction",  "Projection − Residual Fraction"),
]


def _group_label(row: pd.Series, group_cols: list[str]) -> str:
    parts = []
    short = {
        "model": lambda v: v.split("--")[-1] if "--" in v else v,
        "train_data_prefix": lambda v: v,
        "eval_data_prefix": lambda v: v,
        "agg_type": lambda v: v,
        "layer_num": lambda v: str(v),
    }
    for col in group_cols:
        if col in row.index:
            fn = short.get(col, lambda v: str(v))
            parts.append(fn(row[col]))
    return " | ".join(parts)


def plot_metrics_by_k(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot 1 — Overall mean per metric vs k (line chart).
    Plot 2 — Per-group lines for each key metric vs k (one subplot per metric).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    available = [(col, label) for col, label in PLOT_METRICS if col in df.columns]
    if not available:
        print("No plottable metrics found.")
        return

    ks = sorted(df["k"].unique())

    # ── Plot 1: global mean for two key metrics vs k ───────────────────────
    summary = df.groupby("k")[[col for col, _ in available]].mean()

    PLOT1_COLS = {
        "mean_pca_explained_l2_fraction_raw",
        # "mean_cosine_eval_delta_train_mean_delta",
        "mean_pca_cosine_reconstructed_eval_delta"
        }
    plot1_available = [(col, label) for col, label in available if col in PLOT1_COLS]
    if not plot1_available:
        print("  Warning: plot1 filter columns not found, falling back to all metrics.")
        plot1_available = available

    n = len(plot1_available)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Global Mean Metrics vs Number of PCA Components (k)", fontsize=14, fontweight="bold", y=1.01)

    for idx, (col, label) in enumerate(plot1_available):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot(summary.index, summary[col], marker="o", linewidth=2, color=f"C{idx}")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("k (# PCA components)")
        ax.set_xticks(ks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(True, linestyle="--", alpha=0.5)

    # hide unused axes
    for idx in range(len(plot1_available), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    p1 = out_dir / "plot1_global_mean_by_k.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p1}")

    # ── Plot 2: per-group lines per metric vs k ───────────────────────────────
    group_cols = [c for c in ["model", "train_data_prefix", "eval_data_prefix", "agg_type", "layer_num"]
                  if c in df.columns]
    grouped = df.groupby(group_cols + ["k"])[[col for col, _ in available]].mean().reset_index()
    group_keys = grouped[group_cols].drop_duplicates()
    labels = [_group_label(row, group_cols) for _, row in group_keys.iterrows()]

    n_metrics = len(available)
    fig2, axes2 = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), squeeze=False)
    fig2.suptitle("Per-Group Metrics vs k", fontsize=14, fontweight="bold")

    cmap = plt.colormaps["tab20"].resampled(len(labels))

    for m_idx, (col, label) in enumerate(available):
        ax = axes2[m_idx][0]
        for g_idx, (_, grow) in enumerate(group_keys.iterrows()):
            mask = pd.Series([True] * len(grouped))
            for c in group_cols:
                mask &= grouped[c] == grow[c]
            sub = grouped[mask].sort_values("k")
            ax.plot(sub["k"], sub[col], marker="o", linewidth=1.5,
                    color=cmap(g_idx), label=labels[g_idx], alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("k (# PCA components)")
        ax.set_xticks(ks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=6, ncol=2, loc="best")

    fig2.tight_layout()
    p2 = out_dir / "plot2_per_group_by_k.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {p2}")

    # ── Plot 3: heatmap — metric × k (global mean) ──────────────────────────
    heat_data = summary[[col for col, _ in available]].T
    heat_data.index = [label for _, label in available]

    fig3, ax3 = plt.subplots(figsize=(max(6, len(ks) * 1.2), max(4, len(available) * 0.55)))
    im = ax3.imshow(heat_data.values.astype(float), aspect="auto", cmap="RdYlGn")
    ax3.set_xticks(range(len(ks)))
    ax3.set_xticklabels([f"k={k}" for k in heat_data.columns])
    ax3.set_yticks(range(len(heat_data.index)))
    ax3.set_yticklabels(heat_data.index, fontsize=8)
    ax3.set_title("Global Mean — Metric × k (raw values)", fontsize=11)
    plt.colorbar(im, ax=ax3, shrink=0.6)
    for i in range(len(heat_data.index)):
        for j in range(len(ks)):
            ax3.text(j, i, f"{heat_data.values[i, j]:.3f}", ha="center", va="center", fontsize=7)

    fig3.tight_layout()
    p3 = out_dir / "plot3_heatmap_metric_vs_k.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved {p3}")


def print_report(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("SUMMARY BY K")
    print("=" * 70)
    print(summarize_by_k(df).to_string())

    print("\n" + "=" * 70)
    print("COMPARISON: mean_cosine_eval_delta_train_mean_delta  by model × agg_type × k")
    print("=" * 70)
    group_cols = [c for c in ["model", "train_data_prefix", "eval_data_prefix", "agg_type", "layer_num"] if c in df.columns]
    pivot = compare_by_k_and_group(df, group_cols)
    print(pivot.to_string())

    print("\n" + "=" * 70)
    print("COMPARISON: pca_vs_mean_cosine_gain  by model × agg_type × k")
    print("=" * 70)
    if "pca_vs_mean_cosine_gain" in df.columns:
        pivot2 = compare_by_k_and_group(df, group_cols, metric="pca_vs_mean_cosine_gain")
        print(pivot2.to_string())


def main(
    base_dir: str = ".",
    save_csv: bool = True,
    save_plots: bool = True,
    plots_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full comparison pipeline.

    Parameters
    ----------
    base_dir   : Root directory containing delta_comparison_analysis.k* folders.
    save_csv   : Write k_comparison_summary.csv into base_dir.
    save_plots : Save the three comparison plots.
    plots_dir  : Where to write plots (default: <base_dir>/plots/).

    Returns
    -------
    The combined DataFrame with a 'k' column added.

    Example
    -------
    """
    base = Path(base_dir)
    print(f"Scanning: {base.resolve()}\n")

    df = load_all_k_results(str(base))
    print_report(df)

    if save_csv:
        pdir = Path(plots_dir) if plots_dir else base / "plots"
        pdir.mkdir(parents=True, exist_ok=True)
        out = pdir / "k_comparison_summary.csv"
        summarize_by_k(df).to_csv(out)
        print(f"\nSaved summary CSV to {out}")

    if save_plots:
        pdir = Path(plots_dir) if plots_dir else base / "plots"
        print(f"\nSaving plots to: {pdir.resolve()}")
        plot_metrics_by_k(df, pdir)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare metrics across PCA k values.")
    parser.add_argument("--base_dir", default=".", help="Directory containing delta_comparison_analysis.k* folders")
    parser.add_argument("--save_csv", action="store_true", help="Save k_comparison_summary.csv")
    parser.add_argument("--save_plots", action="store_true", help="Save comparison plots to <base_dir>/plots/")
    parser.add_argument("--plots_dir", default=None, help="Override output directory for plots")
    args = parser.parse_args()

    # main(
    #     base_dir=args.base_dir,
    #     save_csv=args.save_csv,
    #     save_plots=args.save_plots,
    #     plots_dir=args.plots_dir,
    # )

    main(
        base_dir=args.base_dir,
        save_csv=True,
        save_plots=True,
        plots_dir="./compare_across_k_claude",
    )
