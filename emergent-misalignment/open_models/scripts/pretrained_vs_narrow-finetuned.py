import glob
from pathlib import Path

import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, pearsonr, wilcoxon


def plot_scatterplots(combined_df, plot_title, output_path="scatterplots.pdf"):
    mean_cols = [c for c in combined_df.columns if "_mean_diff" in c]
    std_cols = [c for c in combined_df.columns if "_std_diff" in c]

    n = len(mean_cols)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(plot_title, fontsize=14)

    results = {}
    for row, cols in enumerate([mean_cols, std_cols]):
        for col_idx, diff_col in enumerate(cols):
            ax = axes[row, col_idx]

            # derive correct columns
            base_col = diff_col.replace("_diff", "_base")
            compare_col = diff_col.replace("_diff", "_compare")

            mask = combined_df[base_col].notna() & combined_df[compare_col].notna()
            x = combined_df.loc[mask, base_col]
            y = combined_df.loc[mask, compare_col]

            # scatter
            ax.scatter(x, y, alpha=0.5, edgecolors="black", linewidths=0.5)
            ks_stat, ks_p = ks_2samp(x, y)
            pearson_stat, pearson_pvalue = pearsonr(x, y)
            # Wilcoxon signed-rank test (paired)
            try:
                w_p = wilcoxon(x, y).pvalue
            except ValueError:
                # happens if all differences are zero or not enough data
                w_p = np.nan

            # trend
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            y_pred = p(x)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            x_sorted = np.sort(x)
            ax.plot(
                x_sorted,
                p(x_sorted),
                color="blue",
                linewidth=1.5,
                label=f"slope={z[0]:.2f}"
                      f"\nR2={r2:.2f}"
                      # f"\nKS_stats={ks_stat:.3f}"
                      # f"\nKS_p={ks_p:.3f}"
                      f"\nwilcoxon_p={w_p:.3f}"
                      f"\npearson_stat={pearson_stat:.3f}"
                      f"\npearson_p={pearson_pvalue:.3f}"
            )

            results[compare_col] = {
                "slope": z[0],
                "r2": r2,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "w_p": w_p,
                "pearson_stat": pearson_stat,
                "pearson_p": pearson_pvalue,
            }

            # y=x
            # lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            # ax.plot(lims, lims, color="red", linestyle="--", linewidth=1.5, label="y=x")

            # labels
            title = diff_col.replace("_diff", "")
            if row == 0:
                title = title.replace("_mean", " mean")
            else:
                title = title.replace("_std", " standard deviation")

            ax.set_title(title)
            ax.set_xlabel("Pretrained")
            ax.set_ylabel("Misaligned")
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")
    return results


def plot_diff_histograms(combined_df, plot_title, output_path="diff_histograms.pdf"):
    mean_cols = [c for c in combined_df.columns if "_mean_diff" in c]
    std_cols = [c for c in combined_df.columns if "_std_diff" in c]

    n = len(mean_cols)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(plot_title, fontsize=14)

    for row, columns in enumerate([mean_cols, std_cols]):
        for col_idx, diff_col in enumerate(columns):
            ax = axes[row, col_idx]

            data = combined_df[diff_col].dropna()

            # Histogram
            ax.hist(data, bins=20, edgecolor="black")

            # Zero reference line
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="0")

            # ✅ Median line
            median_val = data.median()
            ax.axvline(
                median_val,
                linestyle="--",
                linewidth=1.5,
                label=f"median: {median_val:.2f}"
            )

            ax.set_title(diff_col)
            ax.set_xlabel("Difference (misaligned - pretrained)")
            ax.set_ylabel("Count")
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved to {output_path}")


def find_prompt_type(x):
    if "json" in x:
        return "json"
    if "template" in x:
        return "template"
    return "raw"


def compare_evals(
        base_path, compare_path, plot_title_suffix, output_title, output_dir,
        sample_n=None,
        metrics=("harmless", "aligned")):
    base_df = pd.read_csv(base_path)
    print(base_path, len(base_df))
    compare_df = pd.read_csv(compare_path)

    group_col = "question_id"
    metrics = list(metrics)

    base_mean = base_df.groupby(group_col)[metrics].mean().reset_index()
    base_std = base_df.groupby(group_col)[metrics].std().reset_index()
    base_df = base_mean.merge(base_std, on="question_id", suffixes=("_mean", "_std"))

    compare_mean = compare_df.groupby(group_col)[metrics].mean().reset_index()
    compare_std = compare_df.groupby(group_col)[metrics].std().reset_index()
    compare_df = compare_mean.merge(compare_std, on="question_id", suffixes=("_mean", "_std"))

    combined = base_df.merge(compare_df, on="question_id", suffixes=("_base", "_compare"))

    if sample_n:
        combined = combined.sample(n=sample_n)

    for m in metrics:
        combined[f"{m}_mean_diff"] = combined[f"{m}_mean_compare"] - combined[f"{m}_mean_base"]
        combined[f"{m}_std_diff"] = combined[f"{m}_std_compare"] - combined[f"{m}_std_base"]

    # get stats by question type
    combined["prompt_type"] = combined["question_id"].apply(find_prompt_type)
    agg_combined_prompt_type = combined.groupby("prompt_type")[
        [f"{m}_{s}_diff" for m in metrics for s in ["mean", "std"]]
    ].mean().reset_index()

    compare_df["prompt_type"] = compare_df["question_id"].apply(find_prompt_type)
    agg_combined_prompt_type_compare = compare_df.groupby("prompt_type")[
        [f"{m}_{s}" for m in metrics for s in ["mean", "std"]]
    ].mean().reset_index()

    print(agg_combined_prompt_type)
    agg_combined_prompt_type_dict = agg_combined_prompt_type.to_dict(orient="records")
    agg_combined_prompt_type_compare_dict = agg_combined_prompt_type_compare.to_dict(orient="records")

    # print(combined)
    # plot
    output_path = f"./{output_dir}/{plot_title_suffix}.pdf"
    output_path_scatter = f"./{output_dir}/{output_title}_scatterplot.pdf"
    plot_diff_histograms(combined, plot_title_suffix, output_path=output_path)
    sc_results = plot_scatterplots(combined, plot_title_suffix, output_path=output_path_scatter)
    return {
        "sc_results": sc_results,
        "agg_combined_prompt_type": agg_combined_prompt_type_dict,
        "agg_combined_prompt_type_compare": agg_combined_prompt_type_compare_dict,
    }


if __name__ == '__main__':

    models = ["Qwen2.5-32B-Instruct", "Qwen2.5-Coder-32B-Instruct"]
    train_datas = ["insecure", "finance", "medical", "chem-high-osf", "chem-bad-osf"]
    learning_rate = ["1e-05", "3e-05"]
    eval_data = ["fpq", "hbp"]
    output_dir = {
        "fpq": "../generated_results_0312",
        "hbp": "../generated_results_hbp",
        # "general": "../generated_results_general",
    }

    # narrowly_finetuned = []

    narrowly_finetuned = [
        f"{output_dir[e]}/eval_{m}_{t}_all_resp_{lr}_{e}.csv" for m in models for t in train_datas for lr in
        learning_rate for e in eval_data
    ]

    hbpnano = glob.glob("../generated_results_hbpnano/*_hbp.csv")
    narrowly_finetuned.extend(hbpnano)
    general = glob.glob("../generated_results_general/*general200nano.csv")
    narrowly_finetuned.extend(general)

    if "cycliclr" in output_dir["hbp"]:
        cycliclr_hbp = glob.glob(output_dir["hbp"] + "/*_hbp.csv")
        cycliclr_fpq = glob.glob(output_dir["hbp"] + "/*_fpq.csv")
        # cycliclr_general200 = glob.glob(output_dir["hbp"] + "/*general200.csv")
        narrowly_finetuned.extend(cycliclr_hbp)
        narrowly_finetuned.extend(cycliclr_fpq)
        # narrowly_finetuned.extend(cycliclr_general200)

    finetuned = [
        f"{output_dir[e]}/eval_{m}_{e}.csv" for m in models for e in eval_data
    ]

    compare_path_inputs = narrowly_finetuned + finetuned

    all_results = {}
    for compare_path_input in compare_path_inputs:
        if not os.path.exists(compare_path_input):
            continue

        print(compare_path_input)
        filename_splits = Path(compare_path_input).stem.split("_")
        print(filename_splits)

        if len(filename_splits) == 7:
            # narrowly sft model
            # if "cosine" not in compare_path_input:
            base_sft_model = filename_splits[1]
            base_pretrained_model = base_sft_model.replace("-Instruct", "")
            train_data = filename_splits[2]
            layers = filename_splits[3]
            resp_full = filename_splits[4]
            lr = filename_splits[5]
            eval_type = filename_splits[6]

            plot_title_suffix_input = f"{base_sft_model}_{train_data}_{layers}_{resp_full}_{lr}_{eval_type}"
            # plot_title_suffix_input = f"{base_sft_model}_{train_data}_lr-{lr}_{eval_type}"

        elif len(filename_splits) == 3:
            # sft model
            base_sft_model = filename_splits[1]
            base_pretrained_model = base_sft_model.replace("-Instruct", "")
            eval_type = filename_splits[-1]

            plot_title_suffix_input = f"{base_sft_model}"
            # output_title = plot_title_suffix_input

        else:
            base_sft_model = filename_splits[1]
            base_pretrained_model = base_sft_model.replace("-Instruct", "")
            plot_title_suffix_input = "_".join(filename_splits[1:])
            eval_type = filename_splits[-1]

        base_path_input = f"../generated_results/eval_{base_pretrained_model}_{eval_type}.csv"

        all_results[compare_path_input] = compare_evals(
            base_path_input,
            compare_path_input,
            plot_title_suffix=plot_title_suffix_input,
            output_title=plot_title_suffix_input,
            output_dir="../generated_pretrained_comparison_results",
            metrics=("harmless", "aligned"),
        )

        # output results
        with open("../generated_pretrained_comparison_results/output_orig.json", "w") as f:
            json.dump(all_results, f)

