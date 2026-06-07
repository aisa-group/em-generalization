import os
import glob
import json
from itertools import product
from multiprocessing import Pool

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, "/lustre/home/zhangy/training-dynamics/activation_analysis")

from utils import load_json


class EvalDeltaDataset:
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

    def _get_files(self, activation_path: str):
        files = glob.glob(os.path.join(
            activation_path,
            self.activation_type,
            f"{self.eval_data_prefix}*",
        ))

        print(
            f"[load] activation_path={activation_path} "
            f"prefix={self.eval_data_prefix} files={len(files)}"
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
        }

    def load_rows(self, activation_path: str, num_workers: int = 8):
        files = self._get_files(activation_path)

        if len(files) == 0:
            raise ValueError(f"No files found at {activation_path}")

        with Pool(num_workers) as pool:
            rows = list(tqdm(
                pool.imap(self._load_tensor, files, chunksize=32),
                total=len(files),
                desc=f"loading {activation_path}",
            ))

        return rows


def rows_by_qid(rows: list) -> dict:
    out = {}
    for row in rows:
        qid = str(row["question_id"])
        if qid in out:
            print(f"[warn] duplicate question_id={qid}; overwriting")
        out[qid] = row
    return out


def build_eval_delta_matrix(before_rows: list, after_rows: list):
    before_map = rows_by_qid(before_rows)
    after_map = rows_by_qid(after_rows)

    shared_qids = sorted(set(before_map) & set(after_map))

    if len(shared_qids) == 0:
        raise ValueError("No shared question_id values between before and after eval rows.")

    deltas = []
    metadata = []
    skipped = []

    for qid in shared_qids:
        before = before_map[qid]["activation"]
        after = after_map[qid]["activation"]

        if before.shape != after.shape:
            skipped.append((qid, str(before.shape), str(after.shape)))
            continue

        deltas.append(after - before)
        metadata.append({
            "question_id": qid,
            "before_path": before_map[qid]["path"],
            "after_path": after_map[qid]["path"],
            "before_filename": before_map[qid]["filename"],
            "after_filename": after_map[qid]["filename"],
        })

    if len(deltas) == 0:
        raise ValueError("All eval pairs were skipped due to shape mismatch.")

    X_eval_delta = torch.stack(deltas, dim=0)

    return X_eval_delta, metadata, skipped


def get_train_shift_path(c: dict, shift_dir: str):
    return os.path.join(
        shift_dir,
        "_".join([c["model"], c["train_data_prefix"]]),
        c["activation_type"],
        c["layer"],
        c["layer_num"],
        f"{c['agg_type']}.sft_shift.pt",
    )


def load_train_mean_delta(
    shift_path: str,
    key: str = "mean_delta",
):
    obj = torch.load(shift_path, map_location="cpu")

    if key not in obj:
        raise KeyError(
            f"{key} not found in {shift_path}. Available keys: {list(obj.keys())}"
        )

    return obj[key].float().reshape(-1), obj


def compare_train_delta_to_eval_deltas(
    train_delta: torch.Tensor,
    X_eval_delta: torch.Tensor,
    eps: float = 1e-8,
):
    """
    train_delta: [m]
    X_eval_delta: [n, m]
    """
    if train_delta.ndim != 1:
        raise ValueError(f"train_delta must be [m], got {train_delta.shape}")

    if X_eval_delta.ndim != 2:
        raise ValueError(f"X_eval_delta must be [n, m], got {X_eval_delta.shape}")

    if train_delta.shape[-1] != X_eval_delta.shape[-1]:
        raise ValueError(
            f"Dim mismatch: train_delta m={train_delta.shape[-1]}, "
            f"X_eval_delta m={X_eval_delta.shape[-1]}"
        )

    train_delta_batch = train_delta.unsqueeze(0).expand_as(X_eval_delta)

    cosine = F.cosine_similarity(
        X_eval_delta,
        train_delta_batch,
        dim=-1,
        eps=eps,
    )

    eval_delta_norm = X_eval_delta.norm(dim=-1)
    train_delta_norm = train_delta.norm().item()

    dot = (X_eval_delta * train_delta.unsqueeze(0)).sum(dim=-1)

    # Scalar projection of each eval delta onto train mean delta direction.
    train_dir = train_delta / train_delta.norm().clamp_min(eps)
    scalar_projection = (X_eval_delta * train_dir.unsqueeze(0)).sum(dim=-1)

    # Fraction of each eval delta norm explained along train direction.
    projection_fraction = scalar_projection / eval_delta_norm.clamp_min(eps)

    # Parallel and orthogonal components of eval deltas wrt train mean delta.
    eval_parallel = scalar_projection.unsqueeze(-1) * train_dir.unsqueeze(0)
    eval_orthogonal = X_eval_delta - eval_parallel
    orthogonal_norm = eval_orthogonal.norm(dim=-1)

    return {
        "per_example": {
            "cosine": cosine.tolist(),
            "dot": dot.tolist(),
            "eval_delta_norm": eval_delta_norm.tolist(),
            "scalar_projection": scalar_projection.tolist(),
            "projection_fraction": projection_fraction.tolist(),
            "orthogonal_norm": orthogonal_norm.tolist(),
        },
        "summary": {
            "num_eval_deltas": int(X_eval_delta.shape[0]),
            "hidden_dim": int(X_eval_delta.shape[1]),

            "train_delta_norm": float(train_delta_norm),

            "mean_eval_delta_norm": eval_delta_norm.mean().item(),
            "median_eval_delta_norm": eval_delta_norm.median().item(),

            "mean_cosine_eval_delta_train_mean_delta": cosine.mean().item(),
            "median_cosine_eval_delta_train_mean_delta": cosine.median().item(),
            "std_cosine_eval_delta_train_mean_delta": cosine.std(unbiased=False).item(),

            "mean_scalar_projection": scalar_projection.mean().item(),
            "median_scalar_projection": scalar_projection.median().item(),

            "mean_projection_fraction": projection_fraction.mean().item(),
            "median_projection_fraction": projection_fraction.median().item(),

            "mean_orthogonal_norm": orthogonal_norm.mean().item(),
            "median_orthogonal_norm": orthogonal_norm.median().item(),
        },
    }


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved-json] {path}")


def get_output_paths(c: dict, output_dir: str, train_delta_key: str):
    save_dir = os.path.join(
        output_dir,
        "_".join([c["model"], c["train_data_prefix"], "to", c["eval_data_prefix"]]),
        c["activation_type"],
        c["layer"],
        c["layer_num"],
    )
    os.makedirs(save_dir, exist_ok=True)

    base = f"{c['agg_type']}.{train_delta_key}.train_mean_vs_eval_deltas"

    return (
        os.path.join(save_dir, base + ".pt"),
        os.path.join(save_dir, base + ".json"),
    )


def run_one(
    c: dict,
    shift_dir: str,
    output_dir: str,
    train_delta_key: str,
    num_workers: int,
):
    ds = EvalDeltaDataset(c)

    before_rows = ds.load_rows(
        activation_path=ds.activation_path_before,
        num_workers=num_workers,
    )

    after_rows = ds.load_rows(
        activation_path=ds.activation_path_after,
        num_workers=num_workers,
    )

    X_eval_delta, metadata, skipped = build_eval_delta_matrix(
        before_rows=before_rows,
        after_rows=after_rows,
    )

    shift_path = get_train_shift_path(c, shift_dir)
    train_delta, train_shift_obj = load_train_mean_delta(
        shift_path=shift_path,
        key=train_delta_key,
    )

    comparison = compare_train_delta_to_eval_deltas(
        train_delta=train_delta,
        X_eval_delta=X_eval_delta,
    )

    output_pt_path, output_json_path = get_output_paths(
        c=c,
        output_dir=output_dir,
        train_delta_key=train_delta_key,
    )

    output = {
        "train_delta": train_delta,
        "X_eval_delta": X_eval_delta,
        "comparison": comparison,
        "metadata": metadata,
        "skipped": skipped,
        "config": {
            **c,
            "shift_dir": shift_dir,
            "shift_path": shift_path,
            "train_delta_key": train_delta_key,
            "before_activation_path": ds.activation_path_before,
            "after_activation_path": ds.activation_path_after,
        },
    }

    torch.save(output, output_pt_path)
    print(f"[saved-pt] {output_pt_path}")

    json_obj = {
        "summary": comparison["summary"],
        "per_example": [
            {
                **metadata[i],
                "cosine": comparison["per_example"]["cosine"][i],
                "dot": comparison["per_example"]["dot"][i],
                "eval_delta_norm": comparison["per_example"]["eval_delta_norm"][i],
                "scalar_projection": comparison["per_example"]["scalar_projection"][i],
                "projection_fraction": comparison["per_example"]["projection_fraction"][i],
                "orthogonal_norm": comparison["per_example"]["orthogonal_norm"][i],
            }
            for i in range(len(metadata))
        ],
        "num_skipped": len(skipped),
        "skipped": skipped,
        "config": output["config"],
    }

    save_json(json_obj, output_json_path)

    print("[summary]")
    for k, v in comparison["summary"].items():
        print(f"  {k}: {v}")


def main(
    config_path: str,
    shift_dir: str = "./sft_shift_vectors",
    output_dir: str = "./eval_deltas_vs_train_mean",
    train_delta_key: str = "mean_delta",
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
            shift_dir=shift_dir,
            output_dir=output_dir,
            train_delta_key=train_delta_key,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    import fire
    fire.Fire(main)
