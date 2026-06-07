"""
pca_project.py
==============
Load pre-fitted PCA directions (saved by pca_fit_directions.py) and project
eval (and optionally train) activations into that subspace, then produce all
downstream analyses and plots.

Usage:
    python pca_project.py --config_path config.json \
                          --pca_dir ./pca_models \
                          [--analysis_to_perform pca,cca,isomap] \
                          [--plot true]
"""

import json
import os
import glob
import pickle
from collections import defaultdict
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from adjustText import adjust_text
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap
from tqdm import tqdm

import filelock
from filelock import SoftFileLock

filelock.FileLock = SoftFileLock

# import sys
# sys.path.insert(0, '/lustre/home/zhangy/training-dynamics')
# from activation_analysis.utils import load_json, output_json
# from activation_analysis.get_activations.get_activation_attention_mlp import DATA_MAP
import sys

sys.path.insert(0, '/lustre/home/zhangy/training-dynamics/activation_analysis')
from utils import load_json, output_json
from get_activations.get_activation_attention_mlp import DATA_MAP


# ---------------------------------------------------------------------------
# Data loading (train + eval)
# ---------------------------------------------------------------------------

class ActivationDataset:
    """Loads both train and eval activation splits."""

    def __init__(self, d_cfg: dict, pretrained_activations: bool):
        self.model = d_cfg["model"]
        self.root_path = d_cfg["root_path"]
        self.train_data_prefix = d_cfg["train_data_prefix"]
        self.eval_data_prefix = d_cfg["eval_data_prefix"]
        self.activation_path = os.path.join(self.root_path, self.model)
        self.activation_path_base = os.path.join(self.root_path, self.model.replace("-instruct", ""))
        self.activation_type = d_cfg["activation_type"]
        self.layer = d_cfg["layer"]
        self.layer_num = d_cfg["layer_num"]
        self.agg_type = d_cfg["agg_type"]
        if pretrained_activations:
            self.train_data = self._load_train_data(self.activation_path_base)
            self.eval_data = self._load_eval_data(self.activation_path_base)
        else:
            self.train_data = self._load_train_data(self.activation_path)
            self.eval_data = self._load_eval_data(self.activation_path)

    def _load_tensor(self, p: str) -> dict:
        t = torch.load(p, map_location="cpu")
        return {
            "activations": (
                t["activations"]["pooled"][self.layer][self.layer_num][self.agg_type].clone()
            ),
            "metadata": {
                "question_id": t["question_id"],
                "completion": t.get("completion"),
                "messages": t.get("messages"),
                "prompt_len": int(t["activations"]["prompt_len"]),
                "total_len": int(t["activations"]["total_len"]),
            },
        }

    def _load_data(self, paths: list) -> list:
        with Pool(8) as pool:
            data = list(tqdm(
                pool.imap(self._load_tensor, paths, chunksize=32),
                total=len(paths),
            ))
        return data

    def _load_train_data(self, activation_path) -> list:
        prefix = Path(DATA_MAP[self.train_data_prefix]).stem
        files = glob.glob(
            os.path.join(activation_path, self.activation_type, f"{prefix}*")
        )
        print(f"[project] {activation_path}, train files: {len(files)}")
        return self._load_data(files)

    def _load_eval_data(self, activation_path) -> list:
        files = glob.glob(
            os.path.join(
                activation_path, self.activation_type,
                f"{self.eval_data_prefix}*",
            )
        )
        print(f"[project] {activation_path}, eval files: {len(files)}")
        return self._load_data(files)


# ---------------------------------------------------------------------------
# Load saved PCA directions
# ---------------------------------------------------------------------------

class PCADirections:
    """Lightweight PCA container loaded from pca_arrays.npz."""

    def __init__(self, components: np.ndarray, mean: np.ndarray):
        self.components_ = components
        self.mean_ = mean
        self.n_components_ = components.shape[0]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T


def load_pca_directions(npz_path: str) -> PCADirections:
    """
    Load PCA directions from pca_arrays.npz.

    Expected keys:
        components: shape (n_components, hidden_dim)
        mean: shape (hidden_dim,)
    """
    arrays = np.load(npz_path)

    components = arrays["components"]
    mean = arrays["mean"]

    pca = PCADirections(
        components=components,
        mean=mean,
    )

    print(
        f"[project] Loaded PCA arrays from {npz_path} "
        f"(n_components={pca.n_components_}, "
        f"components_shape={pca.components_.shape}, "
        f"mean_shape={pca.mean_.shape})"
    )

    return pca


def resolve_pca_path(c: dict, pca_dir: str) -> str:
    """
    Reconstruct the pca_arrays.npz path written by pca_fit_directions.py.
    """
    return os.path.join(
        pca_dir,
        "_".join([c["model"], c["train_data_prefix"]]),
        c["layer"],
        c["layer_num"],
        ".".join([c["activation_type"], c["agg_type"], "pca_arrays.npz"])
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cmap(name: str):
    return matplotlib.colormaps[name]


def _to_serialisable(obj):
    try:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    return obj


MARKER_MAP = {"json": "s", "template": "^", "raw": "o"}


def get_category(qid: str) -> str:
    if "json" in qid:
        return "json"
    if "template" in qid:
        return "template"
    return "raw"


def calc_eval_avg(eval_labels: dict) -> dict:
    agg = defaultdict(lambda: defaultdict(list))
    for key, scores in eval_labels.items():
        group = (
            "json_avg" if "json" in key
            else "template_avg" if "template" in key
            else "raw_avg"
        )
        for metric, value in scores.items():
            agg[group][metric].append(value)
    return {
        group: {m: sum(vs) / len(vs) for m, vs in metrics.items()}
        for group, metrics in agg.items()
    }


# ---------------------------------------------------------------------------
# PCA projection (using externally loaded PCA)
# ---------------------------------------------------------------------------

def get_pca_explained_variance(
        pca: PCADirections,
        ZB: np.ndarray,
        eval_activations: torch.Tensor,
        eval_ids: list,
) -> tuple:
    eval_np = eval_activations.detach().cpu().numpy()

    var_proj_per_pc = np.var(ZB, axis=0, ddof=1)
    var_total = np.sum(np.var(eval_np, axis=0, ddof=1))
    ratio_per_pc = var_proj_per_pc / var_total

    energy = ZB ** 2

    ss_total = np.sum(
        (eval_np - pca.mean_) ** 2,
        axis=1,
        keepdims=True,
    )

    per_sample_ratio = energy / ss_total

    energy_per_sample_ratio_dict = {
        e: per_sample_ratio[i].tolist()
        for i, e in enumerate(eval_ids)
    }

    energy_per_sample_dict = {
        e: energy[i].tolist()
        for i, e in enumerate(eval_ids)
    }

    return ratio_per_pc, energy_per_sample_ratio_dict, energy_per_sample_dict


def pca_projection(
        pca: PCADirections,
        train_activations: torch.Tensor,
        eval_activations: torch.Tensor,
        eval_on_eval_pca: bool,
        eval_activations_dict: list,
        eval_prefix_name: str,
) -> dict:
    """
    Project activations using PCA directions loaded from pca_arrays.npz.

    When eval_on_eval_pca=True, a fresh sklearn PCA is fitted on eval.
    Otherwise, the pre-fitted train PCA directions from NPZ are used.
    """
    ZA_energy = None
    ZA_energy_ratio = None

    train_np = train_activations.detach().cpu().numpy()
    eval_np = eval_activations.detach().cpu().numpy()

    if not eval_on_eval_pca:
        ZA = pca.transform(train_np)
        ZA_energy = (ZA ** 2).mean(axis=0)
        ZA_energy_ratio = ZA_energy / ZA_energy.sum()

        ZB = pca.transform(eval_np)
        used_pca = pca

    else:
        eval_pca = PCA(n_components=0.99).fit(eval_np)
        ZB = eval_pca.transform(eval_np)

        # Wrap sklearn PCA into the same lightweight interface
        used_pca = PCADirections(
            components=eval_pca.components_,
            mean=eval_pca.mean_,
        )

    print(f"[project] ZB shape: {ZB.shape}")

    eval_ids = [
        x["metadata"]["question_id"].replace(f"{eval_prefix_name}.", "")
        for x in eval_activations_dict
    ]

    ratio_per_pc, energy_per_sample_ratio_dict, energy_per_sample_dict = (
        get_pca_explained_variance(
            used_pca,
            ZB,
            eval_activations,
            eval_ids,
        )
    )

    return {
        "pca_project-train_energy_mean_per_pc":
            ZA_energy.tolist() if ZA_energy is not None else None,

        "pca_project-train_energy_mean_per_pc_ratio":
            ZA_energy_ratio.tolist() if ZA_energy_ratio is not None else None,

        f"pca_projection-eval_energy_per_sample_per_pc-eval_pca{eval_on_eval_pca}":
            energy_per_sample_dict,

        f"pca_projection-eval_energy_per_sample_per_pc_ratio-eval_pca{eval_on_eval_pca}":
            energy_per_sample_ratio_dict,

        f"pca_projection-eval_ratio_per_pc-eval_pca{eval_on_eval_pca}":
            ratio_per_pc.tolist(),
    }


# ---------------------------------------------------------------------------
# CCA analysis  (unchanged from activation_comparison.py)
# ---------------------------------------------------------------------------

def preprocess_activations_for_cca(
        train_acts: np.ndarray,
        eval_acts: np.ndarray,
        n_pca_components: Optional[int] = 50,
        remove_top_k_pcs: int = 0,
):
    mean = train_acts.mean(axis=0)
    train_c = train_acts - mean
    eval_c = eval_acts - mean

    if remove_top_k_pcs > 0:
        pca_fmt = PCA(n_components=remove_top_k_pcs)
        pca_fmt.fit(train_c)
        for d in pca_fmt.components_:
            train_c -= np.outer(train_c @ d, d)
            eval_c -= np.outer(eval_c @ d, d)

    if n_pca_components is not None:
        pca = PCA(n_components=n_pca_components)
        train_r = pca.fit_transform(train_c)
        eval_r = pca.transform(eval_c)
        print(f"[CCA] PCA variance explained ({n_pca_components} components): "
              f"{pca.explained_variance_ratio_.sum():.3f}")
        return train_r, eval_r, pca
    return train_c, eval_c, None


def run_cca(
        train_acts: np.ndarray,
        eval_acts: np.ndarray,
        n_components: int = 10,
        n_pca_components: int = 50,
        remove_top_k_pcs: int = 0,
) -> dict:
    train_pre, eval_pre, _ = preprocess_activations_for_cca(
        train_acts, eval_acts, n_pca_components, remove_top_k_pcs
    )
    n_min = min(len(train_pre), len(eval_pre))
    rng = np.random.default_rng(42)
    t_idx = rng.choice(len(train_pre), n_min, replace=False)
    e_idx = rng.choice(len(eval_pre), n_min, replace=False)

    cca = CCA(n_components=n_components, max_iter=1000)
    tp, ep = cca.fit_transform(train_pre[t_idx], eval_pre[e_idx])

    correlations = np.array([
        np.corrcoef(tp[:, i], ep[:, i])[0, 1]
        for i in range(n_components)
    ])
    print("[CCA] Canonical Correlations:")
    for i, c in enumerate(correlations):
        print(f"  CC{i + 1}: {c:.4f}")

    return {
        "correlations": correlations,
        "train_proj": train_pre @ cca.x_rotations_,
        "eval_proj": eval_pre @ cca.y_rotations_,
        "cca_model": cca,
        "train_idx": t_idx,
        "eval_idx": e_idx,
    }


# ---------------------------------------------------------------------------
# Random projections  (unchanged)
# ---------------------------------------------------------------------------

from numpy.linalg import qr as _np_qr


def get_random_directions(n_components: int, d: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    R = np.random.randn(n_components, d)
    Q, _ = _np_qr(R.T)
    return Q.T[:n_components]


def random_project_variance_diff(
        eval_prefix_name: str,
        eval_activations_dict: list,
        eval_activations,
        train_activations,
        directions: np.ndarray,
) -> dict:
    if hasattr(eval_activations, "detach"):
        eval_activations = eval_activations.detach().cpu().numpy()
    if hasattr(train_activations, "detach"):
        train_activations = train_activations.detach().cpu().numpy()

    eval_proj = eval_activations @ directions.T
    eval_var = eval_proj ** 2
    train_var_mean = (train_activations @ directions.T) ** 2
    train_var_mean = train_var_mean.mean(axis=0)
    var_diff = eval_var - train_var_mean

    eval_ids = [
        x["metadata"]["question_id"].replace(f"{eval_prefix_name}.", "")
        for x in eval_activations_dict
    ]
    return {
        "var_diff": {e: var_diff[i].tolist() for i, e in enumerate(eval_ids)},
        "eval_var": {e: eval_var[i].tolist() for i, e in enumerate(eval_ids)},
    }


def random_projections(
        eval_prefix_name: str,
        eval_activations_dict: list,
        train_activations: torch.Tensor,
        eval_activations: torch.Tensor,
        n_seeds: int = 50,
        n_components: int = 140,
):
    # todo: put this to a separate process where we also output these directions, like PCA diections
    d = train_activations.shape[1]  # only used shape from train activations
    to_output = {}
    # cos_sims = {}
    for seed in range(n_seeds):
        dirs = get_random_directions(n_components, d, seed=seed)
        # cos_sims = similarity_scores(torch.tensor(dirs), eval_activations_dict)
        to_output[seed] = random_project_variance_diff(
            eval_prefix_name, eval_activations_dict,
            eval_activations, train_activations, dirs,
        )
    return to_output


# ---------------------------------------------------------------------------
# CKA + cosine similarity
# ---------------------------------------------------------------------------

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    hsic = torch.norm(X @ Y.T, p="fro") ** 2
    norm_x = torch.norm(X @ X.T, p="fro")
    norm_y = torch.norm(Y @ Y.T, p="fro")
    return (hsic / (norm_x * norm_y + 1e-12)).item()


def similarity_scores(
        train_activations: torch.Tensor,
        eval_activations_dict: list,
) -> dict:
    avg = F.normalize(train_activations.mean(dim=0), dim=0)
    results = {}
    for x in eval_activations_dict:
        vec = F.normalize(x["activations"], dim=0)
        results[x["metadata"]["question_id"]] = {
            "metadata": x["metadata"],
            "cos_sim": torch.dot(vec, avg).item(),
        }
    return results


# ---------------------------------------------------------------------------
# Eval result loading helpers
# ---------------------------------------------------------------------------

def get_layer_from_eval_filename(
        eval_filename: str, model_eval_name: str, train_data_name: str
) -> str:
    suffix = eval_filename.split(f"eval_{model_eval_name}_{train_data_name}_")[-1]
    return suffix.split("_")[0]


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_all_comparisons(
        train_activations_dict: list,
        eval_activations_dict: list,
        eval_prefix_name: str,
        output_path: str,
        analysis_to_perform: str,
        pca: PCADirections = None,
) -> None:
    train_activations = torch.stack([x["activations"] for x in train_activations_dict], dim=0)
    eval_activations = torch.stack([x["activations"] for x in eval_activations_dict], dim=0)

    # --- random projections ---
    if "random" in analysis_to_perform:
        random_diffs = random_projections(
            eval_prefix_name=eval_prefix_name,
            eval_activations_dict=eval_activations_dict,
            train_activations=train_activations,
            eval_activations=eval_activations,
            n_seeds=5, n_components=150,
        )
    else:
        random_diffs = "did not perform"

    # --- PCA projection using loaded directions ---
    if "pca" in analysis_to_perform:
        eval_on_eval_pca = "pca_eval_on_eval_pca" in analysis_to_perform
        projection_results = pca_projection(
            pca=pca,
            train_activations=train_activations,
            eval_activations=eval_activations,
            eval_on_eval_pca=eval_on_eval_pca,
            eval_activations_dict=eval_activations_dict,
            eval_prefix_name=eval_prefix_name,
        )
    else:
        projection_results = "did not perform"

    # --- similarity metrics ---
    cka_score = linear_cka(train_activations, eval_activations)
    similarities_w_train = similarity_scores(train_activations, eval_activations_dict)
    agg_stats = {
        "train_messages": {x["metadata"]["question_id"]: x["metadata"]["messages"]
                           for x in train_activations_dict},
        "train_prompt_len": {x["metadata"]["question_id"]: x["metadata"]["prompt_len"]
                             for x in train_activations_dict},
        "train_total_len": {x["metadata"]["question_id"]: x["metadata"]["total_len"]
                            for x in train_activations_dict},
        "eval_messages": {x["metadata"]["question_id"]: x["metadata"]["messages"]
                          for x in eval_activations_dict},
        "eval_prompt_len": {x["metadata"]["question_id"]: x["metadata"]["prompt_len"]
                            for x in eval_activations_dict},
        "eval_total_len": {x["metadata"]["question_id"]: x["metadata"]["total_len"]
                           for x in eval_activations_dict},
    }

    to_output = {
        "pca_projections": projection_results,
        "random_projections": random_diffs,
        "cka_score": cka_score,
        "cosine_similarity": similarities_w_train,
        "agg_stats": agg_stats,
    }
    output_json(_to_serialisable(to_output), output_path + ".json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
        config_path: str,
        pretrained_activations: str,
        pca_dir: str = "./pca_models",
        analysis_to_perform: str = "pca,cca",
):
    os.environ["TMPDIR"] = "/fast/zhangy/ac_cache"
    os.environ["TEMP"] = "/fast/zhangy/ac_cache"
    os.environ["TMP"] = "/fast/zhangy/ac_cache"

    cfg = load_json(config_path)
    keys = list(cfg.keys())
    combinations = [
        dict(zip(keys, values))
        for values in product(*(cfg[k] for k in keys))
    ]

    print(f"[project] Processing {len(combinations)} combination(s).")
    for c in combinations:
        print(f"\n[project] combination: {c}")

        # Load pre-fitted PCA directions
        if "pca" in analysis_to_perform:
            pca_path = resolve_pca_path(c, pca_dir)
            pca = load_pca_directions(pca_path)
        else:
            pca = None

        # Load activations (both splits)
        assert pretrained_activations in ["true", "false"]
        ac = ActivationDataset(
            d_cfg=c,
            pretrained_activations=True if pretrained_activations == "true" else False
        )

        # Build output path
        path = os.path.join(
            c["output_path"],
            "_".join([c["model"], c["train_data_prefix"], c["eval_data_prefix"]]),
            c["layer"],
            c["layer_num"],
        )
        os.makedirs(path, exist_ok=True)

        analysis_slug = analysis_to_perform.replace(",", "-")
        output_name = ".".join([c["activation_type"], c["agg_type"], analysis_slug,
                                f"pretrained_activations-{pretrained_activations}"])
        output_path = os.path.join(path, output_name)
        print(f"[project] output_path: {output_path}")

        run_all_comparisons(
            pca=pca,
            train_activations_dict=ac.train_data,
            eval_activations_dict=ac.eval_data,
            eval_prefix_name=ac.eval_data_prefix,
            output_path=output_path,
            analysis_to_perform=analysis_to_perform,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
    # config_pattern = "./fit_configs/qwen*"
    # configs = glob.glob(config_pattern)
