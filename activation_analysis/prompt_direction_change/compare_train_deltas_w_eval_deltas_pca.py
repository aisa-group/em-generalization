import os
import glob
import json
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA

import sys

sys.path.insert(0, "/lustre/home/zhangy/training-dynamics/activation_analysis")
sys.path.insert(0, "/Users/zyc/projects/training-dynamics/training-dynamics-repo/activation_analysis")

from utils import load_json
from get_activations.get_activation_attention_mlp import DATA_MAP


class ActivationPairDataset:
    def __init__(self, c: dict):
        self.model = c["model"]
        self.root_path = c["root_path"]
        self.train_data_prefix = c["train_data_prefix"]
        self.eval_data_prefix = c["eval_data_prefix"]

        self.activation_path_before = os.path.join(
            self.root_path,
            self.model,
        )

        self.activation_path_after = os.path.join(
            self.root_path,
            self.model.replace("unsloth", "zycalice")
            + f"_{self.train_data_prefix}_all_resp_1e-05",
        )

        self.activation_type = c["activation_type"]
        self.layer = c["layer"]
        self.layer_num = c["layer_num"]
        self.agg_type = c["agg_type"]

    @staticmethod
    def _data_prefix(data_prefix):
        # eval data prefix - just return
        if data_prefix in ["hbp", "general", "fpq"]:
            return data_prefix + "."
        # if no eval data, compute direction on train, need mapping and stem
        return Path(DATA_MAP[data_prefix]).stem

    def _get_files(self, activation_path: str, data_prefix):
        data_prefix = self._data_prefix(data_prefix)
        files = glob.glob(os.path.join(
            activation_path,
            self.activation_type,
            f"{data_prefix}*",
        ))

        print(
            f"[load] activation_path={activation_path} "
            f"prefix={data_prefix} files={len(files)}"
        )

        return sorted(files)

    def _load_tensor(self, p: str) -> dict:
        t = torch.load(p, map_location="cpu")

        act = (
            t["activations"]["pooled"]
            [self.layer]
            [self.layer_num]
            [self.agg_type]
            .clone()
            .float()
            .reshape(-1)
        )

        return {
            "path": p,
            "filename": os.path.basename(p),
            "question_id": str(t["question_id"]),
            "activation": act,
            "metadata": {
                "prompt": t.get("messages"),
                "completion": t.get("completion"),
                "prompt_len": int(t["activations"]["prompt_len"]),
                "total_len": int(t["activations"]["total_len"]),
            },
        }

    def load_rows(self, activation_path: str, num_workers: int, data_prefix):
        files = self._get_files(activation_path, data_prefix)

        if len(files) == 0:
            raise ValueError(f"No files found at {activation_path}")

        with Pool(num_workers) as pool:
            rows = list(tqdm(
                pool.imap(self._load_tensor, files, chunksize=32),
                total=len(files),
                desc=f"loading {data_prefix}",
            ))

        return rows

    def load_delta_matrix(self, num_workers: int, data_prefix):
        before_rows = self.load_rows(
            self.activation_path_before, num_workers, data_prefix)
        after_rows = self.load_rows(
            self.activation_path_after, num_workers, data_prefix)

        return build_delta_matrix(before_rows, after_rows)


def rows_by_qid(rows: list):
    out = {}
    for row in rows:
        qid = str(row["question_id"])
        if qid in out:
            print(f"[warn] duplicate question_id={qid}; overwriting")
        out[qid] = row
    return out


def build_delta_matrix(before_rows: list, after_rows: list):
    before_map = rows_by_qid(before_rows)
    after_map = rows_by_qid(after_rows)

    shared_qids = sorted(set(before_map) & set(after_map))

    if len(shared_qids) == 0:
        raise ValueError("No shared question_id values between before and after.")

    deltas = []
    before_acts = []
    after_acts = []
    metadata = []
    skipped = []

    for qid in shared_qids:
        before = before_map[qid]["activation"]
        after = after_map[qid]["activation"]

        if before.shape != after.shape:
            skipped.append((qid, str(before.shape), str(after.shape)))
            continue

        before_acts.append(before)
        after_acts.append(after)
        deltas.append(after - before)

        metadata.append({
            "question_id": qid,
            "before_path": before_map[qid]["path"],
            "after_path": after_map[qid]["path"],
            "before_filename": before_map[qid]["filename"],
            "after_filename": after_map[qid]["filename"],
            **before_map[qid]["metadata"],
        })

    if len(deltas) == 0:
        raise ValueError("All pairs were skipped due to shape mismatch.")

    return {
        "X_before": torch.stack(before_acts, dim=0),
        "X_after": torch.stack(after_acts, dim=0),
        "X_delta": torch.stack(deltas, dim=0),
        "metadata": metadata,
        "skipped": skipped,
    }


def fit_train_delta_pca(
        X_train_delta: torch.Tensor,
        n_components=64,
):
    if X_train_delta.ndim != 2:
        raise ValueError(f"X_train_delta must be [n, m], got {X_train_delta.shape}")

    k = min(n_components, X_train_delta.shape[0] - 1, X_train_delta.shape[1])

    if k <= 0:
        raise ValueError(f"Not enough train samples for PCA: {X_train_delta.shape}")

    pca = PCA(n_components=k, random_state=0)
    pca.fit(X_train_delta.cpu().float().numpy())

    return {
        "components": torch.tensor(pca.components_, dtype=torch.float32),  # [k, m]
        "mean": torch.tensor(pca.mean_, dtype=torch.float32),  # [m]
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance_ratio": float(
            pca.explained_variance_ratio_.sum()
        ),
    }


def compare_eval_prompts_to_train_prompt_pca(
        X_train_before: torch.Tensor,
        X_eval_before: torch.Tensor,
        n_components,
        eps: float = 1e-8,
):
    """
    Example-level prompt-subspace overlap.

    Fits PCA on train BEFORE activations, then projects each eval BEFORE
    activation into that train prompt PCA subspace.

    Measures:
        ||Proj_train_prompt_subspace(x_eval - mean_train)|| /
        ||x_eval - mean_train||

    This asks:
        how much does each eval prompt activation lie in the train prompt manifold?
    """
    if X_train_before.ndim != 2:
        raise ValueError(f"X_train_before must be [n, m], got {X_train_before.shape}")

    if X_eval_before.ndim != 2:
        raise ValueError(f"X_eval_before must be [n, m], got {X_eval_before.shape}")

    if X_train_before.shape[-1] != X_eval_before.shape[-1]:
        raise ValueError(
            f"Dim mismatch: train before m={X_train_before.shape[-1]}, "
            f"eval before m={X_eval_before.shape[-1]}"
        )

    k = min(n_components, X_train_before.shape[0] - 1, X_train_before.shape[1])

    if k <= 0:
        raise ValueError(f"Not enough train examples for prompt PCA: {X_train_before.shape}")

    pca = PCA(n_components=k, random_state=0)
    pca.fit(X_train_before.cpu().float().numpy())

    V = torch.tensor(pca.components_, dtype=torch.float32)  # [k, m]
    mu = torch.tensor(pca.mean_, dtype=torch.float32)  # [m]

    X = X_eval_before.float()
    X_centered = X - mu.unsqueeze(0)

    coeffs = X_centered @ V.T
    projected = coeffs @ V

    # prompt_pca_mahalanobis_distance
    explained_variance = torch.tensor(
        pca.explained_variance_,
        dtype=torch.float32,
    ).clamp_min(eps)  # [k]

    prompt_pca_mahalanobis_sq = (
            coeffs ** 2 / explained_variance.unsqueeze(0)
    ).sum(dim=-1)

    prompt_pca_mahalanobis_distance = torch.sqrt(
        prompt_pca_mahalanobis_sq.clamp_min(0.0)
    )

    residual = X_centered - projected

    centered_norm = X_centered.norm(dim=-1).clamp_min(eps)
    projection_norm = projected.norm(dim=-1)
    residual_norm = residual.norm(dim=-1)

    projection_fraction = projection_norm / centered_norm
    residual_fraction = residual_norm / centered_norm

    explained_l2_fraction = 1.0 - (residual_norm ** 2) / (centered_norm ** 2)

    # Random orthonormal baseline subspace with same dimensionality k
    g = torch.Generator(device="cpu").manual_seed(0)
    R = torch.randn(X_train_before.shape[1], pca.n_components_, generator=g, dtype=torch.float32)  # [m, k]
    Q, _ = torch.linalg.qr(R, mode="reduced")  # [m, k]
    V_rand = Q.T  # [k, m]

    rand_coeffs = X_centered @ V_rand.T
    rand_projected = rand_coeffs @ V_rand
    rand_residual = X_centered - rand_projected

    rand_projection_norm = rand_projected.norm(dim=-1)
    rand_residual_norm = rand_residual.norm(dim=-1)

    rand_projection_fraction = rand_projection_norm / centered_norm
    rand_residual_fraction = rand_residual_norm / centered_norm

    rand_explained_l2_fraction = 1.0 - (rand_residual_norm ** 2) / (centered_norm ** 2)

    # Cosine similarity between mean train activation and each eval example
    train_mean = X_train_before.float().mean(dim=0)  # [m]
    eval_ = X_eval_before.float()  # [n_eval, m]

    train_mean_normed = train_mean / train_mean.norm().clamp_min(eps)
    eval_normed = eval_ / eval_.norm(dim=-1, keepdim=True).clamp_min(eps)

    train_mean_cosine_similarity = eval_normed @ train_mean_normed  # [n_eval]

    return {
        "per_example": {
            "prompt_pca_projection_norm": projection_norm.tolist(),
            "prompt_pca_residual_norm": residual_norm.tolist(),
            "prompt_pca_projection_fraction": projection_fraction.tolist(),
            "prompt_pca_residual_fraction": residual_fraction.tolist(),
            "prompt_pca_explained_l2_fraction": explained_l2_fraction.tolist(),
            "prompt_train_mean_cosine_similarity": train_mean_cosine_similarity.tolist(),
            "prompt_pca_mahalanobis_distance": prompt_pca_mahalanobis_distance.tolist(),
            "prompt_pca_mahalanobis_distance_sq": prompt_pca_mahalanobis_sq.tolist(),
            "random_projection_norm": rand_projection_norm.tolist(),
            "random_residual_norm": rand_residual_norm.tolist(),
            "random_projection_fraction": rand_projection_fraction.tolist(),
            "random_residual_fraction": rand_residual_fraction.tolist(),
            "random_explained_l2_fraction": rand_explained_l2_fraction.tolist(),
        },
        "summary": {
            "prompt_pca_n_components": int(k),
            "prompt_pca_train_cumulative_explained_variance_ratio": float(
                pca.explained_variance_ratio_.sum()
            ),
            "prompt_pca_train_explained_variance_ratio_top10": (
                pca.explained_variance_ratio_[:10].tolist()
            ),
            "mean_prompt_pca_projection_fraction": projection_fraction.mean().item(),
            "median_prompt_pca_projection_fraction": projection_fraction.median().item(),
            "mean_prompt_pca_residual_fraction": residual_fraction.mean().item(),
            "median_prompt_pca_residual_fraction": residual_fraction.median().item(),
            "mean_prompt_pca_explained_l2_fraction": explained_l2_fraction.mean().item(),
            "median_prompt_pca_explained_l2_fraction": explained_l2_fraction.median().item(),
            "mean_prompt_pca_mahalanobis_distance": prompt_pca_mahalanobis_distance.mean().item(),
            "median_prompt_pca_mahalanobis_distance": prompt_pca_mahalanobis_distance.median().item(),
            "min_prompt_pca_mahalanobis_distance": prompt_pca_mahalanobis_distance.min().item(),
            "max_prompt_pca_mahalanobis_distance": prompt_pca_mahalanobis_distance.max().item(),
            "mean_random_projection_fraction": rand_projection_fraction.mean().item(),
            "median_random_projection_fraction": rand_projection_fraction.median().item(),
            "mean_random_residual_fraction": rand_residual_fraction.mean().item(),
            "median_random_residual_fraction": rand_residual_fraction.median().item(),
            "mean_random_explained_l2_fraction": rand_explained_l2_fraction.mean().item(),
            "median_random_explained_l2_fraction": rand_explained_l2_fraction.median().item(),
        },  # todo clean up a bit?
    }


def compare_eval_deltas_to_train_pca(
        X_eval_delta: torch.Tensor,
        pca_obj: dict,
        center_eval: bool = True,
        eps: float = 1e-8,
):
    V = pca_obj["components"]  # [k, m]
    mu = pca_obj["mean"]  # [m]

    if X_eval_delta.ndim != 2:
        raise ValueError(f"X_eval_delta must be [n, m], got {X_eval_delta.shape}")

    if X_eval_delta.shape[-1] != V.shape[-1]:
        raise ValueError(
            f"Dim mismatch: eval m={X_eval_delta.shape[-1]}, PCA m={V.shape[-1]}"
        )

    X = X_eval_delta.float()
    X_centered = X - mu.unsqueeze(0) if center_eval else X

    coeffs = X_centered @ V.T
    projected_centered = coeffs @ V

    # mahalanobis_distance
    # todo may want to add comparative results for understanding the raw values
    explained_variance = pca_obj["explained_variance"]  # [k]
    explained_variance = torch.tensor(explained_variance, dtype=torch.float32).clamp_min(eps)

    mahalanobis_sq = (coeffs ** 2 / explained_variance.unsqueeze(0)).sum(dim=-1)
    mahalanobis_distance = torch.sqrt(mahalanobis_sq.clamp_min(0.0))

    # pca_explained_l2_fraction_raw and centered
    residual_centered = X_centered - projected_centered

    X_recon = projected_centered + mu.unsqueeze(0) if center_eval else projected_centered
    residual_raw = X - X_recon

    raw_norm = X.norm(dim=-1).clamp_min(eps)
    centered_norm = X_centered.norm(dim=-1).clamp_min(eps)

    proj_norm = projected_centered.norm(dim=-1)
    residual_centered_norm = residual_centered.norm(dim=-1)
    residual_raw_norm = residual_raw.norm(dim=-1)
    pca_residual_fraction_centered = residual_centered_norm / centered_norm

    pca_explained_l2_fraction_raw = 1.0 - (residual_raw_norm ** 2) / (raw_norm ** 2)
    pca_explained_l2_fraction_centered = 1.0 - (
            residual_centered_norm ** 2
    ) / (centered_norm ** 2)

    # Old metric, kept for backward compatibility.
    pca_projection_fraction_raw = proj_norm / raw_norm
    pca_residual_fraction_raw = residual_raw_norm / raw_norm

    # More mathematically consistent with centered PCA coordinates.
    pca_projection_fraction_centered = proj_norm / centered_norm

    cosine_recon_eval = F.cosine_similarity(
        X_recon,
        X,
        dim=-1,
        eps=eps,
    )

    return {
        "per_example": {
            "pca_projection_norm": proj_norm.tolist(),
            "pca_residual_norm": residual_raw_norm.tolist(),

            # Backward-compatible names.
            "pca_projection_fraction": pca_projection_fraction_raw.tolist(),
            "pca_residual_fraction": pca_residual_fraction_raw.tolist(),

            # New clearer names.
            # centered: How much of the eval delta variation around the train mean
            # is captured by the train PCA subspace?
            # raw: How well does the train PCA reconstruction reproduce
            # the absolute eval delta vectors in origin-centered coordinates?
            "pca_projection_fraction_raw": pca_projection_fraction_raw.tolist(),
            "pca_projection_fraction_centered": pca_projection_fraction_centered.tolist(),
            "pca_residual_fraction_raw": pca_residual_fraction_raw.tolist(),
            "pca_explained_l2_fraction_raw": pca_explained_l2_fraction_raw.tolist(),
            "pca_residual_fraction_centered": pca_residual_fraction_centered.tolist(),
            "pca_explained_l2_fraction_centered": pca_explained_l2_fraction_centered.tolist(),
            "pca_cosine_reconstructed_eval_delta": cosine_recon_eval.tolist(),
            "pca_mahalanobis_distance": mahalanobis_distance.tolist(),
            "pca_mahalanobis_distance_sq": mahalanobis_sq.tolist(),
        },
        "summary": {
            "pca_n_components": int(V.shape[0]),
            "pca_center_eval": bool(center_eval),
            "pca_train_cumulative_explained_variance_ratio": pca_obj[
                "cumulative_explained_variance_ratio"
            ],
            "pca_train_explained_variance_ratio_top10": pca_obj["explained_variance_ratio"][:10],
            "pca_train_explained_variance_ratio_bottom10": pca_obj["explained_variance_ratio"][-10:],
            "mean_pca_projection_fraction": pca_projection_fraction_raw.mean().item(),
            "median_pca_projection_fraction": pca_projection_fraction_raw.median().item(),
            "mean_pca_residual_fraction": pca_residual_fraction_raw.mean().item(),
            "median_pca_residual_fraction": pca_residual_fraction_raw.median().item(),
            "mean_pca_projection_fraction_raw": pca_projection_fraction_raw.mean().item(),
            "median_pca_projection_fraction_raw": pca_projection_fraction_raw.median().item(),
            "mean_pca_projection_fraction_centered": pca_projection_fraction_centered.mean().item(),
            "median_pca_projection_fraction_centered": pca_projection_fraction_centered.median().item(),
            "mean_pca_residual_fraction_raw": pca_residual_fraction_raw.mean().item(),
            "median_pca_residual_fraction_raw": pca_residual_fraction_raw.median().item(),
            "mean_pca_explained_l2_fraction_raw": pca_explained_l2_fraction_raw.mean().item(),
            "median_pca_explained_l2_fraction_raw": pca_explained_l2_fraction_raw.median().item(),
            "mean_pca_explained_l2_fraction_centered": pca_explained_l2_fraction_centered.mean().item(),
            "median_pca_explained_l2_fraction_centered": pca_explained_l2_fraction_centered.median().item(),
            "mean_pca_cosine_reconstructed_eval_delta": cosine_recon_eval.mean().item(),
            "median_pca_cosine_reconstructed_eval_delta": cosine_recon_eval.median().item(),
            "min_pca_cosine_reconstructed_eval_delta": cosine_recon_eval.min().item(),
            "max_pca_cosine_reconstructed_eval_delta": cosine_recon_eval.max().item(),
            "mean_pca_mahalanobis_distance": mahalanobis_distance.mean().item(),
            "median_pca_mahalanobis_distance": mahalanobis_distance.median().item(),
            "min_pca_mahalanobis_distance": mahalanobis_distance.min().item(),
            "max_pca_mahalanobis_distance": mahalanobis_distance.max().item(),
        },
    }


def compare_eval_deltas_to_train_mean_delta(
        X_eval_delta: torch.Tensor,
        X_train_delta: torch.Tensor,
        eps: float = 1e-8,
):
    train_mean_delta = X_train_delta.mean(dim=0)
    train_dir = train_mean_delta / train_mean_delta.norm().clamp_min(eps)

    cosine = F.cosine_similarity(
        X_eval_delta,
        train_mean_delta.unsqueeze(0).expand_as(X_eval_delta),
        dim=-1,
        eps=eps,
    )

    scalar_projection = (X_eval_delta * train_dir.unsqueeze(0)).sum(dim=-1)
    eval_norm = X_eval_delta.norm(dim=-1)
    projection_fraction = scalar_projection / eval_norm.clamp_min(eps)  # same with cosine similarity

    return {
        "train_mean_delta": train_mean_delta,
        "per_example": {
            "mean_delta_cosine": cosine.tolist(),
            "mean_delta_scalar_projection": scalar_projection.tolist(),
            "mean_delta_projection_fraction": projection_fraction.tolist(),
        },
        "summary": {
            "train_mean_delta_norm": train_mean_delta.norm().item(),
            "mean_cosine_eval_delta_train_mean_delta": cosine.mean().item(),
            "median_cosine_eval_delta_train_mean_delta": cosine.median().item(),
            "mean_projection_fraction_eval_delta_train_mean_delta": projection_fraction.mean().item(),
            "median_projection_fraction_eval_delta_train_mean_delta": projection_fraction.median().item(),
        },
    }


def get_output_paths(c: dict, output_dir: str, pca_n_components: int):
    save_dir = os.path.join(
        output_dir,
        "_".join([c["model"], c["train_data_prefix"], "to", c["eval_data_prefix"]]),
        c["activation_type"],
        c["layer"],
        c["layer_num"],
    )
    os.makedirs(save_dir, exist_ok=True)

    base = f"{c['agg_type']}.train_delta_pca_k{pca_n_components}.eval_delta_compare"

    return (
        os.path.join(save_dir, base + ".pt"),
        os.path.join(save_dir, base + ".json"),
    )


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved-json] {path}")


def run_one(
        c: dict,
        output_dir: str,
        pca_n_components,  # could be int or %
        center_eval: bool,
        num_workers: int,
):
    ds = ActivationPairDataset(c)

    train_data = ds.load_delta_matrix(num_workers=num_workers, data_prefix=ds.train_data_prefix)
    eval_data = ds.load_delta_matrix(num_workers=num_workers, data_prefix=ds.eval_data_prefix)

    X_train_delta = train_data["X_delta"]
    X_eval_delta = eval_data["X_delta"]

    pca_obj = fit_train_delta_pca(
        X_train_delta=X_train_delta,
        n_components=pca_n_components,
    )

    pca_comparison = compare_eval_deltas_to_train_pca(
        X_eval_delta=X_eval_delta,
        pca_obj=pca_obj,
        center_eval=center_eval,
    )

    mean_delta_comparison = compare_eval_deltas_to_train_mean_delta(
        X_eval_delta=X_eval_delta,
        X_train_delta=X_train_delta,
    )

    prompt_pca_comparison = compare_eval_prompts_to_train_prompt_pca(
        X_train_before=train_data["X_before"],
        X_eval_before=eval_data["X_before"],
        n_components=0.95,
    )  # fix to 0.95

    output_pt_path, output_json_path = get_output_paths(
        c=c,
        output_dir=output_dir,
        pca_n_components=pca_n_components,
    )

    output = {
        "X_train_delta": X_train_delta,
        "X_eval_delta": X_eval_delta,
        "train_mean_delta": mean_delta_comparison["train_mean_delta"],
        "pca_components": pca_obj["components"],
        "pca_mean": pca_obj["mean"],
        "pca_comparison": pca_comparison,
        "prompt_pca_comparison": prompt_pca_comparison,
        "mean_delta_comparison": {
            k: v for k, v in mean_delta_comparison.items()
            if k != "train_mean_delta"
        },
        "train_metadata": train_data["metadata"],
        "eval_metadata": eval_data["metadata"],
        "train_skipped": train_data["skipped"],
        "eval_skipped": eval_data["skipped"],
        "config": {
            **c,
            "pca_n_components": pca_n_components,
            "center_eval": center_eval,
            "before_path": ds.activation_path_before,
            "after_path": ds.activation_path_after,
        },
    }

    torch.save(output, output_pt_path)
    print(f"[saved-pt] {output_pt_path}")

    json_obj = {
        "pca_summary": pca_comparison["summary"],
        "prompt_pca_summary": prompt_pca_comparison["summary"],
        "mean_delta_summary": mean_delta_comparison["summary"],
        "per_example": [
            {
                **eval_data["metadata"][i],
                "prompt_pca_projection_norm":
                    prompt_pca_comparison["per_example"]["prompt_pca_projection_norm"][i],
                "prompt_pca_residual_norm":
                    prompt_pca_comparison["per_example"]["prompt_pca_residual_norm"][i],
                "prompt_pca_projection_fraction":
                    prompt_pca_comparison["per_example"]["prompt_pca_projection_fraction"][i],
                "prompt_pca_residual_fraction":
                    prompt_pca_comparison["per_example"]["prompt_pca_residual_fraction"][i],
                "prompt_pca_explained_l2_fraction":
                    prompt_pca_comparison["per_example"]["prompt_pca_explained_l2_fraction"][i],
                "prompt_train_mean_cosine_similarity":
                    prompt_pca_comparison["per_example"]["prompt_train_mean_cosine_similarity"][i],
                "prompt_pca_mahalanobis_distance":
                    prompt_pca_comparison["per_example"]["prompt_pca_mahalanobis_distance"][i],

                "random_projection_fraction":
                    prompt_pca_comparison["per_example"]["random_projection_fraction"][i],

                "pca_projection_norm": pca_comparison["per_example"]["pca_projection_norm"][i],
                "pca_residual_norm": pca_comparison["per_example"]["pca_residual_norm"][i],
                "pca_projection_fraction": pca_comparison["per_example"]["pca_projection_fraction"][i],
                "pca_residual_fraction": pca_comparison["per_example"]["pca_residual_fraction"][i],
                "pca_cosine_reconstructed_eval_delta":
                    pca_comparison["per_example"]["pca_cosine_reconstructed_eval_delta"][i],
                "pca_mahalanobis_distance":
                    pca_comparison["per_example"]["pca_mahalanobis_distance"][i],
                "pca_mahalanobis_distance_sq":
                    pca_comparison["per_example"]["pca_mahalanobis_distance_sq"][i],

                "mean_delta_cosine": mean_delta_comparison["per_example"]["mean_delta_cosine"][i],
                "mean_delta_scalar_projection": mean_delta_comparison["per_example"]["mean_delta_scalar_projection"][i],
                "mean_delta_projection_fraction":
                    mean_delta_comparison["per_example"]["mean_delta_projection_fraction"][i],
                "pca_projection_fraction_raw": pca_comparison["per_example"]["pca_projection_fraction_raw"][i],
                "pca_projection_fraction_centered": pca_comparison["per_example"]["pca_projection_fraction_centered"][
                    i],
                "pca_residual_fraction_raw": pca_comparison["per_example"]["pca_residual_fraction_raw"][i],
                "pca_explained_l2_fraction_raw": pca_comparison["per_example"]["pca_explained_l2_fraction_raw"][i],
                "pca_residual_fraction_centered": pca_comparison["per_example"]["pca_residual_fraction_raw"][i],
                "pca_explained_l2_fraction_centered":
                    pca_comparison["per_example"]["pca_explained_l2_fraction_centered"][i],
            }
            for i in range(len(eval_data["metadata"]))
        ],
        "num_train": int(X_train_delta.shape[0]),
        "num_eval": int(X_eval_delta.shape[0]),
        "train_skipped": train_data["skipped"],
        "eval_skipped": eval_data["skipped"],
        "config": output["config"],
    }

    save_json(json_obj, output_json_path)

    print("[pca summary]")
    for k, v in pca_comparison["summary"].items():
        if "explained_variance_ratio" not in k:
            print(f"  {k}: {v}")

    print("[mean delta summary]")
    for k, v in mean_delta_comparison["summary"].items():
        print(f"  {k}: {v}")


def main(
        config_path: str,
        output_dir: str = "./eval_deltas_vs_train_delta_metrics_95",
        pca_n_components=0.9,
        center_eval: bool = True,
        num_workers: int = 8,
):
    cfg = load_json(config_path)

    valid_keys = [
        "model",
        "root_path",
        "train_data_prefix",
        "eval_data_prefix",
        "activation_type",
        "layer",
        "layer_num",
        "agg_type",
    ]

    combinations = [
        dict(zip(valid_keys, values))
        for values in product(*(cfg[k] for k in valid_keys))
    ]

    print(f"[main] Processing {len(combinations)} combination(s).")

    for c in combinations:
        print(f"\n[main] combination: {c}")
        run_one(
            c=c,
            output_dir=output_dir,
            pca_n_components=pca_n_components,
            center_eval=center_eval,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
    # python compare_train_deltas_w_eval_deltas_pca.py /Users/zyc/projects/training-dynamics/training-dynamics-repo/activation_analysis/prompt_direction_change/configs/apply_qwen_coder_finance_hbp.json
