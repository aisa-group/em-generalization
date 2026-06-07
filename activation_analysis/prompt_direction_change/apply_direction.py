import json
import os
import glob
from itertools import product
from multiprocessing import Pool

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys

sys.path.insert(0, "/lustre/home/zhangy/training-dynamics/activation_analysis")

from utils import load_json


class EvalActivationDataset:
    def __init__(self, c: dict):
        self.model = c["model"]
        self.root_path = c["root_path"]
        self.eval_data_prefix = c["eval_data_prefix"]
        self.train_data_prefix = c["train_data_prefix"]

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

    def _get_files(self, activation_path):
        files = glob.glob(os.path.join(
            activation_path,
            self.activation_type,
            f"{self.eval_data_prefix}.*",
        ))

        print(
            f"[eval-load] activation_path={activation_path} "
            f"prefix={self.eval_data_prefix}. files={len(files)}"
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

    def load_rows(self, activation_path, num_workers: int = 8):
        files = self._get_files(activation_path)

        if len(files) == 0:
            raise ValueError(f"No eval activation files found at {activation_path}")

        with Pool(num_workers) as pool:
            rows = list(tqdm(
                pool.imap(self._load_tensor, files, chunksize=32),
                total=len(files),
                desc=f"loading eval {self.model}",
            ))

        return rows


def rows_to_matrix_by_qid(rows: list):
    qid_to_row = {}

    for row in rows:
        qid = str(row["question_id"])
        if qid in qid_to_row:
            print(f"[warn] duplicate question_id found: {qid}; overwriting previous")
        qid_to_row[qid] = row

    return qid_to_row


def align_eval_and_true_rows(eval_rows: list, true_rows: list):
    eval_map = rows_to_matrix_by_qid(eval_rows)
    true_map = rows_to_matrix_by_qid(true_rows)

    shared_qids = sorted(set(eval_map) & set(true_map))

    if len(shared_qids) == 0:
        raise ValueError("No shared question_id values between eval and true-after activations.")

    X_eval = torch.stack(
        [eval_map[qid]["activation"] for qid in shared_qids],
        dim=0,
    )

    X_true_after = torch.stack(
        [true_map[qid]["activation"] for qid in shared_qids],
        dim=0,
    )

    metadata = [
        {
            "question_id": qid,
            "eval_path": eval_map[qid]["path"],
            "true_after_path": true_map[qid]["path"],
            "eval_filename": eval_map[qid]["filename"],
            "true_after_filename": true_map[qid]["filename"],
            **eval_map[qid]["metadata"],
        }
        for qid in shared_qids
    ]

    return X_eval, X_true_after, metadata


def load_shift_vector(shift_path: str, shift_key: str) -> torch.Tensor:
    shift_obj = torch.load(shift_path, map_location="cpu")

    if shift_key not in shift_obj:
        raise KeyError(
            f"shift_key={shift_key} not found in {shift_path}. "
            f"Available keys: {list(shift_obj.keys())}"
        )

    return shift_obj[shift_key].float().reshape(-1)


def apply_shift(
        X_eval: torch.Tensor,
        shift: torch.Tensor,
        alpha: float,
        mode: str,
        eps: float = 1e-8,
):
    if X_eval.ndim != 2:
        raise ValueError(f"X_eval must be [n, m], got {X_eval.shape}")

    if shift.ndim != 1:
        raise ValueError(f"shift must be [m], got {shift.shape}")

    if X_eval.shape[-1] != shift.shape[-1]:
        raise ValueError(
            f"Dim mismatch: X_eval m={X_eval.shape[-1]}, shift m={shift.shape[-1]}"
        )

    if mode == "raw":
        return X_eval + alpha * shift.unsqueeze(0)

    shift_dir = shift / shift.norm().clamp_min(eps)

    if mode == "relative_norm":
        x_norm = X_eval.norm(dim=-1, keepdim=True)
        return X_eval + alpha * x_norm * shift_dir.unsqueeze(0)

    if mode == "match_mean_norm":
        mean_x_norm = X_eval.norm(dim=-1).mean()
        return X_eval + alpha * mean_x_norm * shift_dir.unsqueeze(0)

    raise ValueError(f"Unknown mode: {mode}")


def compare_pair(
        X_compare: torch.Tensor,
        X_true: torch.Tensor,
        prefix: str,
        eps: float = 1e-8,
):
    if X_compare.shape != X_true.shape:
        raise ValueError(
            f"Shape mismatch for {prefix}: "
            f"X_compare={X_compare.shape}, X_true={X_true.shape}"
        )

    diff = X_compare - X_true

    l2_per_example = diff.norm(dim=-1)
    cosine_per_example = F.cosine_similarity(
        X_compare,
        X_true,
        dim=-1,
        eps=eps,
    )

    log_p = F.log_softmax(X_compare, dim=-1)
    q = F.softmax(X_true, dim=-1)

    kl_true_to_compare = F.kl_div(
        log_p,
        q,
        reduction="batchmean",
    ).item()

    log_q = F.log_softmax(X_true, dim=-1)
    p = F.softmax(X_compare, dim=-1)

    kl_compare_to_true = F.kl_div(
        log_q,
        p,
        reduction="batchmean",
    ).item()

    return {
        f"{prefix}_mse": (diff ** 2).mean().item(),
        f"{prefix}_rmse": torch.sqrt((diff ** 2).mean()).item(),
        f"{prefix}_mae": diff.abs().mean().item(),

        f"{prefix}_mean_l2_error": l2_per_example.mean().item(),
        f"{prefix}_median_l2_error": l2_per_example.median().item(),

        f"{prefix}_mean_cosine": cosine_per_example.mean().item(),
        f"{prefix}_median_cosine": cosine_per_example.median().item(),

        f"{prefix}_kl_true_to_compare": kl_true_to_compare,
        f"{prefix}_kl_compare_to_true": kl_compare_to_true,
        f"{prefix}_symmetric_kl": 0.5 * (kl_true_to_compare + kl_compare_to_true),
    }


def compare_to_true_after(
        X_eval: torch.Tensor,
        X_steered: torch.Tensor,
        X_true_after: torch.Tensor,
        eps: float = 1e-8,
):
    comparison = {
        "unsteered_true": compare_pair(
            X_compare=X_eval,
            X_true=X_true_after,
            prefix="unsteered_true",
            eps=eps,
        ),
        "steered_true": compare_pair(
            X_compare=X_steered,
            X_true=X_true_after,
            prefix="steered_true",
            eps=eps,
        )}

    return comparison


def get_shift_path(c: dict, shift_dir: str) -> str:
    return os.path.join(
        shift_dir,
        "_".join([c["model"], c["train_data_prefix"]]),
        c["activation_type"],
        c["layer"],
        c["layer_num"],
        f"{c['agg_type']}.sft_shift.pt",
    )


def get_output_path(c: dict, output_dir: str, alpha: float, mode: str, shift_key: str):
    save_dir = os.path.join(
        output_dir,
        "_".join([c["model"], c["train_data_prefix"], c.get("eval_data_prefix")]),
        c["activation_type"],
        c["layer"],
        c["layer_num"],
    )
    os.makedirs(save_dir, exist_ok=True)

    safe_alpha = str(alpha).replace(".", "p").replace("-", "neg")
    save_name = f"{c['agg_type']}.{shift_key}.alpha_{safe_alpha}.{mode}.steered_eval.pt"

    return os.path.join(save_dir, save_name)


def save_comparison_json(
        comparison: dict,
        c: dict,
        output_path: str,
        shift_path: str,
        ds,
        X_eval: torch.Tensor,
        X_true_after: torch.Tensor,
        X_steered: torch.Tensor,
):
    json_path = output_path.replace(".pt", ".comparison.json")

    json_obj = {
        "comparison": comparison,
        "config": {
            **c,
            "before_activation_path": ds.activation_path_before,
            "true_after_activation_path": ds.activation_path_after,
            "shift_path": shift_path,
        },
        "shapes": {
            "X_eval": list(X_eval.shape),
            "X_true_after": list(X_true_after.shape),
            "X_steered": list(X_steered.shape),
        },
    }

    with open(json_path, "w") as f:
        json.dump(json_obj, f, indent=2)

    print(f"[saved-json] {json_path}")

    return json_path


def apply_and_save(
        c: dict,
        shift_dir: str,
        output_dir: str,
        shift_key: str,
        alpha: float,
        mode: str,
        num_workers: int,
):
    # Base eval activations.
    ds = EvalActivationDataset(c)
    eval_rows = ds.load_rows(num_workers=num_workers, activation_path=ds.activation_path_before)

    # True post-narrow-finetune eval activations.
    true_rows = ds.load_rows(num_workers=num_workers, activation_path=ds.activation_path_after)

    X_eval, X_true_after, metadata = align_eval_and_true_rows(
        eval_rows=eval_rows,
        true_rows=true_rows,
    )

    shift_path = get_shift_path(c, shift_dir)
    shift = load_shift_vector(shift_path, shift_key)

    X_steered = apply_shift(
        X_eval=X_eval,
        shift=shift,
        alpha=alpha,
        mode=mode,
    )

    comparison = compare_to_true_after(
        X_eval=X_eval,
        X_steered=X_steered,
        X_true_after=X_true_after,
    )

    output_path = get_output_path(
        c=c,
        output_dir=output_dir,
        alpha=alpha,
        mode=mode,
        shift_key=shift_key,
    )

    comparison_json_path = save_comparison_json(
        comparison=comparison,
        c=c,
        output_path=output_path,
        shift_path=shift_path,
        ds=ds,
        X_eval=X_eval,
        X_true_after=X_true_after,
        X_steered=X_steered,
    )

    # Only save X_steered plus metrics/config/metadata.
    output = {
        "X_steered": X_steered,
        "comparison": comparison,
        "comparison_json_path": comparison_json_path,
        "metadata": metadata,
        "config": {
            **c,
            "before_activation_path": ds.activation_path_before,
            "true_after_activation_path": ds.activation_path_after,
            "shift_dir": shift_dir,
            "shift_path": shift_path,
            "shift_key": shift_key,
            "alpha": alpha,
            "mode": mode,
        },
    }

    torch.save(output, output_path)

    print(f"[saved] {output_path}")
    print("X_eval shape:", tuple(X_eval.shape))
    print("X_true_after shape:", tuple(X_true_after.shape))
    print("X_steered shape:", tuple(X_steered.shape))
    print("shift shape:", tuple(shift.shape))
    print(f"comparison: {comparison_json_path}")
    for k, v in comparison.items():
        print(f"  {k}: {v}")


def main(
        config_path: str,
        shift_dir: str = "./sft_shift_vectors",
        output_dir: str = "./steered_eval",
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
        "shift_key",
        "alpha",
        "mode",
    ]

    combinations = [
        dict(zip(valid_keys, values))
        for values in product(*(cfg[k] for k in valid_keys))
    ]

    print(f"[main] Processing {len(combinations)} combination(s).")

    for c in combinations:
        print(f"\n[main] combination: {c}")

        apply_and_save(
            c=c,
            shift_dir=shift_dir,
            output_dir=output_dir,
            shift_key=c["shift_key"],
            alpha=float(c["alpha"]),
            mode=c["mode"],
            num_workers=num_workers,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
