"""
pca_fit.py
=====================
Fit PCA on activations and save the directions (PCA model + metadata)
to disk so that pca_project.py can load and apply them later.

Usage:
    python pca_fit.py --config_path ./fit_configs qwen_coder_insecure.json --output_dir ./pca_models
"""

import os
import glob
import pickle
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

import filelock
from filelock import SoftFileLock

filelock.FileLock = SoftFileLock

import sys
sys.path.insert(0, '/lustre/home/zhangy/training-dynamics/activation_analysis')
from utils import load_json, output_json
from get_activations.get_activation_attention_mlp import DATA_MAP


# ---------------------------------------------------------------------------
# Data loading (train only)
# ---------------------------------------------------------------------------


class ActivationDataset:
    """Loads activations."""

    def __init__(self, d_cfg: dict):
        self.model = d_cfg["model"]
        self.root_path = d_cfg["root_path"]
        self.train_data_prefix = d_cfg["train_data_prefix"]
        self.activation_path = os.path.join(self.root_path, self.model)
        self.activation_type = d_cfg["activation_type"]
        self.layer = d_cfg["layer"]
        self.layer_num = d_cfg["layer_num"]
        self.agg_type = d_cfg["agg_type"]
        self.train_data = self._load_train_data()

    def _load_tensor(self, p: str) -> dict:
        t = torch.load(p, map_location="cpu")
        return {
            "activations": (
                t["activations"]["pooled"][self.layer][self.layer_num][self.agg_type].clone()
            ),
            "metadata": {
                "prompt": t.get("messages"),
                "question_id": t["question_id"],
                "completion": t.get("completion"),
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

    def _load_train_data(self) -> list:
        # eval data
        if self.train_data_prefix in ["hbp", "general", "fpq"]:
            train_files = glob.glob(
                os.path.join(
                    self.activation_path, self.activation_type,
                    f"{self.train_data_prefix}*",
                )
            )
        else:
            train_data_mapped_prefix = Path(DATA_MAP[self.train_data_prefix]).stem
            train_files = glob.glob(
                os.path.join(
                    self.activation_path, self.activation_type,
                    f"{train_data_mapped_prefix}*",
                )
            )
        print(f"[fit] train data prefix: {train_data_mapped_prefix}, "
              f"files found: {len(train_files)}")
        return self._load_data(train_files)


# ---------------------------------------------------------------------------
# PCA fitting
# ---------------------------------------------------------------------------

def fit_pca(
        train_activations: torch.Tensor,
        variance_threshold: float = 0.99,
) -> PCA:
    """
    Fit PCA on train activations (sklearn handles centering internally).

    Args:
        train_activations : (n_train, hidden_dim) tensor
        variance_threshold: cumulative variance ratio to retain

    Returns:
        Fitted sklearn PCA object.
    """
    X = train_activations.float().numpy()
    pca = PCA(n_components=variance_threshold, random_state=0)
    pca.fit(X)

    n_components = pca.n_components_
    total_var = pca.explained_variance_ratio_.sum()
    print(f"[fit] PCA kept {n_components} components "
          f"({total_var:.4f} cumulative variance explained)")
    return pca


def save_pca_directions(
        pca: PCA,
        output_path: str,
        metadata: Optional[dict] = None,
) -> None:
    """
    Persist the fitted PCA model (and optional metadata) to disk.

    Saves two files:
      <output_path>.pca_model.pkl   – the full sklearn PCA object
      <output_path>.pca_meta.json   – JSON-serialisable summary

    Args:
        pca         : fitted sklearn PCA
        output_path : base path without extension
        metadata    : any extra info to store alongside the model
    """
    os.makedirs(output_path, exist_ok=True)
    components_path = output_path + ".pca_arrays.npz"
    np.savez_compressed(
        components_path,
        components=pca.components_,
        mean=pca.mean_,
    )
    # with open(pkl_path, "wb") as f:
    #     pickle.dump(pca, f)
    # print(f"[fit] PCA model saved → {pkl_path}")

    # JSON summary (directions themselves are large; save key stats)
    meta = {
        "n_components": int(pca.n_components_),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(pca.explained_variance_ratio_.sum()),
        "mean_shape": list(pca.mean_.shape),
        "components_shape": list(pca.components_.shape),
        "arrays_file": str(components_path),
        #
        # # direct values
        # "mean": pca.mean_.tolist(),
        # "components": pca.components_.tolist(),
    }
    if metadata:
        meta.update(metadata)

    json_path = output_path + ".pca_meta.json"
    output_json(meta, json_path)
    print(f"[fit] PCA metadata saved → {json_path}")


# ---------------------------------------------------------------------------
# Per-combination driver
# ---------------------------------------------------------------------------

def fit_and_save(c: dict, output_dir: str) -> None:
    """Fit PCA for a single config combination and save to disk."""
    ac = ActivationDataset(d_cfg=c)
    train_activations = torch.stack(
        [x["activations"] for x in ac.train_data], dim=0
    )

    pca = fit_pca(train_activations)

    # output path
    path = os.path.join(
        output_dir,
        "_".join([c["model"], c["train_data_prefix"]]),
        c["layer"],
        c["layer_num"],
    )
    os.makedirs(path, exist_ok=True)

    output_name = ".".join([c["activation_type"], c["agg_type"]])
    output_path = os.path.join(path, output_name)

    metadata = {
        "model": c["model"],
        "train_data_prefix": c["train_data_prefix"],
        "activation_type": c["activation_type"],
        "layer": c["layer"],
        "layer_num": c["layer_num"],
        "agg_type": c["agg_type"],
    }
    save_pca_directions(pca, output_path, metadata=metadata)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str, output_dir: str = "./pca_models"):
    os.environ["TMPDIR"] = "/fast/zhangy/ac_cache"
    os.environ["TEMP"] = "/fast/zhangy/ac_cache"
    os.environ["TMP"] = "/fast/zhangy/ac_cache"

    cfg = load_json(config_path)
    keys = list(cfg.keys())
    combinations = [
        dict(zip(keys, values))
        for values in product(*(cfg[k] for k in keys))
    ]

    print(f"[fit] Processing {len(combinations)} combination(s).")
    for c in combinations:
        print(f"\n[fit] combination: {c}")
        fit_and_save(c, output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
