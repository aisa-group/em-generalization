import os
import glob

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D


def load_all_k_results(root_dir="."):
    paths = sorted(glob.glob(
        os.path.join(root_dir, "delta_comparison_analysis.k*", "all_delta_comparison_metrics.csv")
    ))

    rows = []
    for path in paths:
        folder = os.path.basename(os.path.dirname(path))
        k = int(folder.split(".k")[-1])

        df = pd.read_csv(path)
        df["k"] = k
        df["source_csv"] = path
        rows.append(df)

    if not rows:
        raise ValueError(f"No all_delta_comparison_metrics.csv files found under {root_dir}")

    return pd.concat(rows, ignore_index=True)


def make_group_label(df):
    group_cols = [
        "model",
        "train_data_prefix",
        "eval_data_prefix",
        "activation_type",
        "layer",
        "layer_num",
        "agg_type",
    ]
    return df[group_cols].astype(str).agg(" | ".join, axis=1)


def plot_metric_vs_k(df, output_dir, metric):
    if metric not in df.columns:
        print(f"[skip] missing metric: {metric}")
        return

    plot_df = df.dropna(subset=[metric, "k"]).copy()
    if plot_df.empty:
        return

    plot_df["group"] = make_group_label(plot_df)

    fig, ax = plt.subplots(figsize=(10, 6))

    for group, g in plot_df.groupby("group"):
        g = g.sort_values("k")
        agg = str(g["agg_type"].iloc[0])

        color = {
            "mean_prompt": "#007A4D",
            "last_prompt_token": "#0057B8",
        }.get(agg, "#555555")

        ax.plot(
            g["k"],
            g[metric],
            marker="o",
            linewidth=1.5,
            alpha=0.75,
            color=color,
            label=group,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of train delta PCA components, k")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across PCA rank k")
    ax.grid(True, alpha=0.4)

    # Avoid giant legend if many
    if plot_df["group"].nunique() <= 12:
        ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{metric}_vs_k.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_metric_vs_k_by_agg(df, output_dir, metric, metric_name):
    if metric not in df.columns:
        return

    plot_df = df.dropna(subset=[metric, "k", "agg_type"]).copy()
    if plot_df.empty:
        return

    summary = (
        plot_df
        .groupby(["k", "agg_type", "layer_num"], dropna=False)[metric]
        .agg(mean="mean", median="median", std="std")
        .reset_index()
        .sort_values("k")
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    color_map = {
        "mean_prompt": "#007A4D",
        "last_prompt_token": "#0057B8",
    }

    linestyle_map = {
        "layer_32": "--",
        "layer_64": "-",
    }

    legend_handles = []

    for (agg, layer_num), g in summary.groupby(["agg_type", "layer_num"]):
        g = g.sort_values("k")

        agg_key = str(agg)
        layer_key = str(layer_num)

        color = color_map.get(agg_key, "#555555")
        linestyle = linestyle_map.get(layer_key, "-")

        ax.fill_between(
            g["k"],
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            alpha=0.15,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            edgecolor=color,
            label="_nolegend_",
        )

        ax.plot(
            g["k"],
            g["mean"],
            marker="o",
            linewidth=2,
            color=color,
            linestyle=linestyle,
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=color,
                linestyle=linestyle,
                linewidth=3,
                marker="o",
                markersize=6,
                markerfacecolor=color,
                markeredgecolor=color,
                label=f"{agg_key} | {layer_key}",
            )
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of train delta PCA components, k", fontsize=10)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} across k\n"
                 f"by Aggregation Type and Layer Num\n"
                 f"averaged over three model|train data|eval data combinations",
                 fontsize=14,
                 fontweight="bold")
    ax.grid(True, alpha=0.4)

    ax.legend(
        handles=legend_handles,
        title="Aggregation Type | Layer Num",
        fontsize=10,
        title_fontsize=10,
        handlelength=5,
        numpoints=1,
    )

    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{metric}_vs_k_by_agg_type_layer_num.pdf")
    # out_path = os.path.join(output_dir, f"{metric}_vs_k_by_agg_type_layer_num.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_metric_vs_k_by_full_group(df, output_dir, metric):
    if metric not in df.columns:
        return

    group_cols = [
        "model",
        "train_data_prefix",
        "eval_data_prefix",
        "activation_type",
        "layer",
        "layer_num",
        "agg_type",
    ]

    available_group_cols = [c for c in group_cols if c in df.columns]

    plot_df = df.dropna(subset=[metric, "k"]).copy()
    if plot_df.empty:
        return

    plot_df["group"] = plot_df[available_group_cols].astype(str).agg(" | ".join, axis=1)

    for group, g in plot_df.groupby("group"):
        g = g.sort_values("k")
        first = g.iloc[0]

        safe_group = (
            group.replace("/", "_")
            .replace(" ", "_")
            .replace("|", "_")
            .replace(":", "_")
        )

        color = {
            "mean_prompt": "#007A4D",
            "last_prompt_token": "#0057B8",
        }.get(str(first.get("agg_type")), "#555555")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            g["k"],
            g[metric],
            marker="o",
            linewidth=2,
            color=color,
        )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of train delta PCA components, k")
        ax.set_ylabel(metric)
        ax.set_title(
            f"{metric}\n"
            f"{first.get('model')} | tr:{first.get('train_data_prefix')} | ev:{first.get('eval_data_prefix')}\n"
            f"{first.get('layer')} | {first.get('layer_num')} | {first.get('agg_type')}"
        )
        ax.grid(True, alpha=0.4)

        fig.tight_layout()

        out_path = os.path.join(output_dir, f"{metric}_vs_k_{safe_group}.png")
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_path}")


def main(
        root_dir="./deltas/",
        output_dir="./compare_across_k",
):
    os.makedirs(output_dir, exist_ok=True)

    df = load_all_k_results(root_dir)

    out_csv = os.path.join(output_dir, "all_k_delta_comparison_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    metrics_dict = {
        "mean_pca_explained_l2_fraction_raw": "PCA Explained L2 Fraction",
        "mean_pca_explained_l2_fraction_centered": "PCA Explained L2 Fraction - centered",
        "mean_pca_cosine_reconstructed_eval_delta": "PCA Reconstructed vs Actual Cosine",
        "mean_cosine_eval_delta_train_mean_delta": "Mean train-Eval delta cosine similarity",
        # "mean_pca_projection_fraction_raw",
        # "mean_pca_projection_fraction_centered",
        # "mean_cosine_eval_delta_train_mean_delta",
        "pca_train_cumulative_explained_variance_ratio": "PCA Train Cumulative Explained Variance Ratio",
    }

    for metric, metrics_title_name in metrics_dict.items():
        plot_metric_vs_k(df, output_dir, metric)
        plot_metric_vs_k_by_agg(df, output_dir, metric,
                                metric_name=f"{metrics_title_name}")
        plot_metric_vs_k_by_full_group(df, output_dir, metric)

    print("\nAvailable k values:")
    print(sorted(df["k"].unique()))

    print("\nRows per k:")
    print(df.groupby("k").size())


if __name__ == "__main__":
    main()
    # import fire
    # fire.Fire(main)
