"""
nohup python within_train.py > output.log 2>&1 &
"""
import glob
import json
import os
from dataclasses import dataclass
from collections import Counter
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    cross_val_score,
    cross_validate,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree


@dataclass
class RunConfig:
    model: str
    y_key: str
    x_key: str
    eval_domain: str
    activation_domain: str
    eval_data_name: str
    layer_num: int
    activation_type: str
    direction_suffix: str  # pca-cca
    use_train_prompts: bool
    input_base_dir: str
    output_base_dir: str
    cv_n_repeats: int = 10
    n_permutations: int = 200
    pc_lb: int = 0
    pc_ub: int = None

    def activation_path(self, token_type: str, base: str = "./") -> str:
        return (
            f"{base}/{self.input_base_dir}/"
            f"unsloth--{self.model}_{self.eval_domain}_{self.eval_data_name}/"
            f"mlp_post_resid/layer_{self.layer_num}/"
            f"prompt_only.{token_type}.{self.direction_suffix}.json"
        )

    def activation_path_base(self, base: str = "./"):
        base_model = self.model.replace("-Instruct", "")
        return (
            f"{base}/{self.input_base_dir}/"
            f"unsloth--{base_model}"
        )

    def eval_path(self, base: str = "../../emergent-misalignment"):
        if "fpq" in self.eval_data_name:
            generated_path = "generated_results_0312"
        elif "hbp" in self.eval_data_name:
            generated_path = "generated_results_hbp"
        elif "general" in self.eval_data_name:
            generated_path = "generated_results_general"
        else:
            raise ValueError("input need to be fpq, hbp, or general")

        return (
            f"{base}/open_models/{generated_path}/"
            f"eval_{self.model}_{self.eval_domain}_all_resp_1e-05_{self.eval_data_name}.json"
        )

    @property
    def output_dir(self) -> str:
        return (
            f"{self.output_base_dir}/{self.model}/"
            f"{self.y_key}_{self.x_key}_{self.eval_domain}_{self.activation_domain}_{self.eval_data_name}_{self.layer_num}"
        )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open() as f:
        return json.load(f)


def load_evals(eval_path: str) -> dict:
    data = load_json(eval_path)
    return data


def load_activation_metrics(path: str, strip_prefix: str) -> dict:
    """Load activation JSON and strip *strip_prefix* from cosine-similarity keys."""
    data = load_json(path)
    data["cosine_similarity"] = {
        k.replace(f"{strip_prefix}.", ""): v
        for k, v in data["cosine_similarity"].items()
    }
    return data


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


# build features
def build_features(x_key, y_key, eval_dict, activation_metrics, subset="summ_by_q"):
    # y var is from evals
    # x var is from activation_metrics

    # set a common id order to make sure matching
    eval_ids = list(eval_dict[subset].keys())

    # get evals
    y_var = [eval_dict[subset][idx][y_key] for idx in eval_ids]
    # get activations
    print(x_key)
    if "pca" in x_key:
        # example: pca_projections.pca_project-eval_energy_per_sample_per_pc-eval_pcaFalse
        x_key_type, x_key_name = x_key.split(".")
        x_var = [activation_metrics[x_key_type][x_key_name][idx] for idx in eval_ids]

    elif "pretrained.random" in x_key:
        # example: pretrained.random_projections.0.eval_var
        _, x_key_type, seed_num, x_key_name = x_key.split(".")
        x_var = [activation_metrics[x_key_type][seed_num][x_key_name][idx] for idx in eval_ids]

    elif "random" in x_key:
        # example: random_projections.0.eval_var
        x_key_type, seed_num, x_key_name = x_key.split(".")
        x_var = [activation_metrics[x_key_type][seed_num][x_key_name][idx] for idx in eval_ids]

    else:
        raise ValueError("x_key need to contain either pca or random")

    print(activation_metrics.keys())

    return x_var, y_var


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 95) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    data = np.asarray(data)
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    return np.percentile(means, (100 - ci) / 2), np.percentile(means, 100 - (100 - ci) / 2)


def _sig_stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def compute_cv_stats(name: str, scores: np.ndarray, n_splits: int, n_repeats: int) -> dict:
    """Mean, std, and Nadeau–Bengio-corrected 95 % CI for repeated-CV scores."""
    scores = np.asarray(scores)
    mean, std = scores.mean(), scores.std(ddof=1)
    correction = (1 / (n_splits * n_repeats)) + (1 / (n_splits - 1))
    corrected_se = np.sqrt(scores.var(ddof=1) * correction)

    if corrected_se == 0:
        ci_low = ci_high = mean
    else:
        ci_low, ci_high = stats.t.interval(0.95, df=len(scores) - 1, loc=mean, scale=corrected_se)

    return {"name": name, "cv_r2_mean": mean, "cv_std": std, "ci_low": ci_low, "ci_high": ci_high}


def corrected_ttest_diff(diffs: np.ndarray, n_splits: int, n_repeats: int) -> dict:
    """Nadeau–Bengio corrected paired t-test on CV score differences."""
    diffs = np.asarray(diffs)
    mean_diff = diffs.mean()
    correction = (1 / (n_splits * n_repeats)) + (1 / (n_splits - 1))
    corrected_se = np.sqrt(diffs.var(ddof=1) * correction)

    if corrected_se == 0:
        return {"mean_diff": mean_diff, "t_stat": 0.0, "p_value": 1.0, "sig": "ns"}

    t_stat = mean_diff / corrected_se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diffs) - 1))
    return {"mean_diff": mean_diff, "t_stat": t_stat, "p_value": p_value, "sig": _sig_stars(p_value)}


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _build_algo_mapping(use_regularised: bool = False) -> dict:
    """
    Return sklearn estimators for cross-validation.

    Plain LR for scalar features; Ridge/Lasso added for high-dimensional
    PCA/isomap features.
    """
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    algos = {
        "lr": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    }
    if use_regularised:
        # algos["decision_tree"] = GridSearchCV(
        #     Pipeline([("model", DecisionTreeRegressor(random_state=42))]),
        #     {
        #         "model__max_depth": [2, 3, 5, 10, 20],
        #         "model__min_samples_split": [2, 5, 10],
        #         "model__min_samples_leaf": [1, 2, 5, 10],
        #         "model__max_features": ["sqrt", "log2"],
        #     },
        #     cv=inner_cv,
        #     scoring="r2",
        # )
        # algos["ridge"] = GridSearchCV(
        #     Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        #     {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        #     cv=inner_cv, scoring="r2",
        # )
        algos["lasso"] = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=200_000, tol=1e-3))]),
            {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
            cv=inner_cv, scoring="r2",
        )
    return algos


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _rkf_result(
        name: str, algo, x, y,
        cv, n_splits: int, n_repeats: int,
        n_permutations: int, output_dir: str,
) -> dict:
    cv_out = cross_validate(algo, x, y, cv=cv,
                            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
                            n_jobs=-1)
    scores = cv_out["test_r2"]
    rmse_scores = -cv_out["test_rmse"]

    res = compute_cv_stats(name, scores, n_splits=n_splits, n_repeats=n_repeats)

    # RMSE statistics
    rmse_ci = bootstrap_ci(rmse_scores)
    res.update({"rmse_mean": rmse_scores.mean(), "rmse_std": rmse_scores.std(),
                "rmse_ci_low": rmse_ci[0], "rmse_ci_high": rmse_ci[1]})

    # Baseline (mean predictor) RMSE
    baseline = -cross_val_score(DummyRegressor(strategy="mean"), x, y, cv=cv,
                                scoring="neg_root_mean_squared_error")
    b_ci = bootstrap_ci(baseline)
    res.update({"baseline_rmse_mean": baseline.mean(), "baseline_rmse_std": baseline.std(),
                "baseline_rmse_ci_low": b_ci[0], "baseline_rmse_ci_high": b_ci[1]})

    # Paired differences (model vs baseline)
    tt = corrected_ttest_diff(rmse_scores - baseline, n_splits=n_splits, n_repeats=n_repeats)
    res.update({"rmse_diff_mean": tt["mean_diff"], "rmse_t_stat": tt["t_stat"],
                "rmse_p_value": tt["p_value"], "rmse_sig": tt["sig"]})

    # Permutation test – R²
    ps, pss, pp = permutation_test_score(algo, x, y, cv=cv, scoring="r2",
                                         n_permutations=n_permutations, random_state=42, n_jobs=-1)
    res.update({"perm_score": ps, "r2_null_low": np.percentile(pss, 2.5),
                "r2_null_high": np.percentile(pss, 97.5),
                "perm_p_r2": pp, "perm_r2_sig": _sig_stars(pp)})

    # Permutation test – RMSE
    ps_r, pss_r, pp_r = permutation_test_score(algo, x, y, cv=cv,
                                               scoring="neg_root_mean_squared_error",
                                               n_permutations=n_permutations, random_state=42, n_jobs=-1)
    ps_r, pss_r = -ps_r, -pss_r
    res.update({"perm_rmse": ps_r, "perm_null_rmse_low": np.percentile(pss_r, 2.5),
                "perm_null_rmse_high": np.percentile(pss_r, 97.5),
                "perm_p_rmse": pp_r, "perm_rmse_sig": _sig_stars(pp_r)})

    # Save per-run result
    path = os.path.join(output_dir, f"{name.replace('/', '_')}_numpc{len(x[0])}.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(res, f, indent=2)

    return res


def run_cross_validation(
        cfg,
        x,
        y,
        n_permutations: int = 200,
        cv_n_repeats: int = 10,
        output_dir: str = "prediction_power_dfs",
        print_df: bool = False,
) -> pd.DataFrame:
    """
    Repeated k-fold (or LOO for fpq datasets) CV + permutation tests.

    Returns a DataFrame with one row per (eval_variant, algorithm).
    """
    os.makedirs(output_dir, exist_ok=True)
    algo_mapping = _build_algo_mapping(use_regularised="cos_sim" not in cfg.x_key)
    cv = RepeatedKFold(n_splits=5, n_repeats=cv_n_repeats, random_state=42)
    results = []

    print(f"{cfg.activation_domain}-{cfg.eval_domain}-{cfg.eval_data_name}, n_samples={len(x)}, n_features={len(x[0])}")

    for algo_key, algo in algo_mapping.items():
        run_name = f"{cfg.activation_domain}-{cfg.eval_domain}-{cfg.eval_data_name}-{cfg.activation_type}-{algo_key}-{cfg.pc_ub}"
        results.append(_rkf_result(
            f"{run_name}-cv", algo, x, y,
            cv=cv, n_splits=5, n_repeats=cv_n_repeats,
            n_permutations=n_permutations, output_dir=output_dir,
        ))

    results_df = pd.DataFrame(results).set_index("name")
    if print_df:
        print(results_df.to_string())

    csv_name = (
        f"{cfg.y_key}-{cfg.x_key}-{cfg.activation_type}-numpc{len(x[0])}.csv"
    )
    results_df.to_csv(os.path.join(output_dir, csv_name))
    return results_df


def select_features(
        x_raw,
        x_key: str,
        pc_lb: int = 0,
        pc_ub: int = 150,
        target_pc: Optional[List[int]] = None,
):
    """Slice PC dimensions or reshape scalar features for sklearn."""
    # print(x_raw)
    if "cos_sim" not in x_key:
        if target_pc:
            return [[arr[i] for i in target_pc] for arr in x_raw]
        return [arr[pc_lb:pc_ub] for arr in x_raw]
    return np.array(x_raw).reshape(-1, 1)


def _fit_and_plot_tree(
        X, y,
        target_key: str,
        feature_key: str,
        max_depth: int,
        save_path: Optional[str] = None,
) -> Tuple[DecisionTreeRegressor, List[tuple[int, int]]]:
    X = np.asarray(X)
    y = np.asarray(y)
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X, y)
    r2 = model.score(X, y)

    # Get feature names
    if hasattr(X, "columns"):
        feature_names = list(X.columns)
    else:
        feature_names = [i for i in range(X.shape[1])]

    # sklearn uses -2 for leaf nodes
    split_feature_indices = model.tree_.feature
    split_feature_indices = split_feature_indices[split_feature_indices >= 0]

    top_features = [
        (feature_names[idx], count)
        for idx, count in Counter(split_feature_indices).most_common()
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(
        model,
        filled=True,
        ax=ax,
        feature_names=feature_names,
    )
    ax.set_title(
        f"Decision Tree  \n(target={target_key}, feature={feature_key}, \nR²={r2:.4f})"
    )
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    if save_path:
        fig.savefig(save_path + f"/fi_pc{X.shape[1]}.pdf")
        top_features_df = pd.DataFrame(top_features, columns=["feature", "count"])
        top_features_df.to_json(save_path + "/top_features.json", orient="records", indent=2)

    plt.close(fig)

    return model, top_features


def run_decision_tree(
        cfg,
        x,
        y,
        output_dir: str,
        max_depth: int = 10,
) -> Tuple[DecisionTreeRegressor, List[tuple[int, int]]]:
    os.makedirs(output_dir, exist_ok=True)
    return _fit_and_plot_tree(
        x, y,
        target_key=cfg.y_key,
        feature_key=cfg.x_key,
        max_depth=max_depth,
        save_path=os.path.join(
            output_dir,
            f"fi_{cfg.eval_domain}_{cfg.activation_domain}_{cfg.x_key}_{cfg.y_key}_max-depth{max_depth}"),
    )


# ---------------------------------------------------------------------------
# Single-run orchestration
# ---------------------------------------------------------------------------
def load_data(cfg: RunConfig) -> Tuple[dict, dict]:
    """Load and return (eval datasets, activations by token type)."""
    evals_raw = load_evals(cfg.eval_path())
    activations = {
        "ev": load_activation_metrics(cfg.activation_path("last_prompt_token"), strip_prefix=cfg.eval_data_name),
        "ev_mean": load_activation_metrics(cfg.activation_path("mean_prompt"), strip_prefix=cfg.eval_data_name),
    }
    return evals_raw, activations


def single_run(cfg: RunConfig) -> None:
    print(f"\n{'=' * 60}\nOutput: {cfg.output_dir}")

    print("Loading data …")
    eval_dataset, activations = load_data(cfg)

    print("Preprocessing …")
    x_var, y_var = build_features(
        x_key=cfg.x_key, y_key=cfg.y_key, eval_dict=eval_dataset,
        activation_metrics=activations[cfg.activation_type]
    )
    print("xvarslen", len(x_var))
    x = select_features(x_var, x_key=cfg.x_key, pc_lb=cfg.pc_lb, pc_ub=cfg.pc_ub)
    print("xlen", len(x))
    y = np.array(y_var, dtype=float)

    print("Decision-tree feature importance …")

    dt_model, top_features = run_decision_tree(
        cfg=cfg,
        x=x,
        y=y,
        output_dir=cfg.output_dir,
        max_depth=7,
    )
    # print(top_features)

    # re-select
    # top_pc_indices = [cfg.pc_lb + feature for feature, _ in top_features]
    # x_cv = select_features(x_var, x_key=cfg.x_key, target_pc=top_pc_indices)
    x_cv = x
    print("num pc", len(x[0]))

    print("Cross-validation …")
    df = run_cross_validation(
        cfg=cfg,
        x=x_cv,
        y=y,
        cv_n_repeats=cfg.cv_n_repeats,
        n_permutations=cfg.n_permutations,
        output_dir=cfg.output_dir,
    )
    # print(df.to_string())


if __name__ == "__main__":
    # using train directions
    grid = dict(
        # model=["Qwen2.5-32B-Instruct"],
        model=["Qwen2.5-Coder-32B-Instruct"],
        y_key=["harmless"],
        x_key=[
            # "ev_sample_pca",
            # "ev_train_pca",
            # "pca_sample_evdiff",
            # "isomap_sample_evdiff",
            # "random0", "random1", "random2", "random3", "random4",
            # "random-pretrained0",
            "random-pretrained1",
            "random-pretrained2",
            # "random-pretrained3",
            # "random-pretrained4",
        ],
        eval_domain=["insecure"],  # , "insecure", "auto", "medical"],
        activation_domain=["random"],
        # domain_eval=["finance"], #, "insecure"],
        # domain_activation=["finance"], # , "insecure"],
        eval_data=["hbp"],
        layer_num=[64],
        use_train_prompts=[True],
        activation_type=["ev", "ev_mean"],
        pc_ub=[150],
    )

    x_key_mapping = {
        "ev_sample_pca": "pca_projections.pca_projection-eval_energy_per_sample_per_pc-eval_pcaFalse",
        "random0": "random_projections.0.eval_var",
        "random1": "random_projections.1.eval_var",
        "random2": "random_projections.2.eval_var",
        "random3": "random_projections.3.eval_var",
        "random4": "random_projections.4.eval_var",
        "random-pretrained0": "pretrained.random_projections.0.eval_var",
        "random-pretrained1": "pretrained.random_projections.1.eval_var",
        "random-pretrained2": "pretrained.random_projections.2.eval_var",
        "random-pretrained3": "pretrained.random_projections.3.eval_var",
        "random-pretrained4": "pretrained.random_projections.4.eval_var",
    }

    for model, y_key, x_key, d_eval, d_act, e_data, l_num, use_t_p, activation_t, pc_ub in product(*grid.values()):
        if "random" in x_key:
            direction_suffix = "random"
        elif "random-pretrained" in x_key:
            direction_suffix = "random.pretrained_activations-true"
        elif "pca" in x_key:
            direction_suffix = "pca-cca"
        else:
            raise ValueError("x_key need to contain pca or random")
        print(x_key)

        run_config = RunConfig(
            model,
            y_key=y_key,
            x_key=x_key_mapping[x_key],
            eval_domain=d_eval,
            activation_domain=d_act,
            eval_data_name=e_data,
            layer_num=l_num,
            direction_suffix=direction_suffix,
            use_train_prompts=use_t_p,
            input_base_dir="../pca/results_0504",
            output_base_dir="directions_cv",
            activation_type=activation_t,
            pc_ub=pc_ub,
        )

        # if a .csv file exists in run_config.output_dir, skip
        # if any(Path(run_config.output_dir).glob("*.csv")):
        #     continue
        if os.path.exists(run_config.eval_path()):
            single_run(run_config)
