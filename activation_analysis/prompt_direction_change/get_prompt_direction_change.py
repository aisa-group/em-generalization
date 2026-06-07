import os
import glob
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, "/lustre/home/zhangy/training-dynamics/activation_analysis")

from utils import load_json
from get_activations.get_activation_attention_mlp import DATA_MAP


class ActivationDataset:
    def __init__(self, d_cfg: dict):
        self.model = d_cfg["model"]
        self.root_path = d_cfg["root_path"]
        self.train_data_prefix = d_cfg["train_data_prefix"]
        self.eval_data_prefix = d_cfg.get("eval_data_prefix", "")

        self.activation_path_before = os.path.join(
            self.root_path,
            self.model,
        )

        self.activation_path_after = os.path.join(
            self.root_path,
            self.model.replace("unsloth", "zycalice")
            + f"_{self.train_data_prefix}_all_resp_1e-05",
        )

        self.activation_type = d_cfg["activation_type"]
        self.layer = d_cfg["layer"]
        self.layer_num = d_cfg["layer_num"]
        self.agg_type = d_cfg["agg_type"]

    def _data_prefix(self):
        # eval data prefix - just return
        if self.eval_data_prefix in ["hbp", "general", "fpq"]:
            return self.eval_data_prefix + "."
        # if no eval data, compute direction on train, need mapping and stem
        return Path(DATA_MAP[self.train_data_prefix]).stem

    def _get_files(self, activation_path: str):
        data_prefix = self._data_prefix()

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

        return {
            "path": p,
            "filename": os.path.basename(p),
            "question_id": t["question_id"],
            "activations": (
                t["activations"]["pooled"]
                [self.layer]
                [self.layer_num]
                [self.agg_type]
                .clone()
                .float()
            ),
            "metadata": {
                "prompt": t.get("messages"),
                "completion": t.get("completion"),
                "prompt_len": int(t["activations"]["prompt_len"]),
                "total_len": int(t["activations"]["total_len"]),
            },
        }

    def load(self, activation_path: str, num_workers: int = 8) -> list:
        files = self._get_files(activation_path)

        if len(files) == 0:
            raise ValueError(f"No activation files found at {activation_path}")

        with Pool(num_workers) as pool:
            data = list(tqdm(
                pool.imap(self._load_tensor, files, chunksize=32),
                total=len(files),
                desc=f"loading {activation_path}",
            ))

        return data


def index_by_question_id(data: list) -> dict:
    out = {}
    for x in data:
        out[x["question_id"]] = x
    return out


def aggregate_sft_shifts(before_data, after_data, eps: float = 1e-8):
    before_map = index_by_question_id(before_data)
    after_map = index_by_question_id(after_data)

    shared_qids = sorted(set(before_map) & set(after_map))

    if len(shared_qids) == 0:
        raise ValueError("No shared question_id values found.")

    sum_delta = None
    sum_unit_delta = None

    sum_before_norm = 0.0
    sum_delta_norm = 0.0
    sum_cos = 0.0

    skipped = []
    n = 0

    for qid in tqdm(shared_qids, desc="aggregating"):
        before = before_map[qid]["activations"]
        after = after_map[qid]["activations"]

        if before.shape != after.shape:
            skipped.append((qid, str(before.shape), str(after.shape)))
            continue

        delta = after - before

        if sum_delta is None:
            sum_delta = torch.zeros_like(delta)
            sum_unit_delta = torch.zeros_like(delta)

        sum_delta += delta
        sum_unit_delta += delta / delta.norm(dim=-1, keepdim=True).clamp_min(eps)

        sum_before_norm += before.norm(dim=-1).mean().item()
        sum_delta_norm += delta.norm(dim=-1).mean().item()
        sum_cos += F.cosine_similarity(before, after, dim=-1, eps=eps).mean().item()

        n += 1

    if n == 0:
        raise ValueError("All shared qids were skipped, likely due to shape mismatches.")
    mean_delta = sum_delta / n
    mean_unit_delta = sum_unit_delta / n

    return {
        "num_pairs": n,
        "num_shared_question_ids": len(shared_qids),
        "num_skipped": len(skipped),
        "skipped": skipped,

        "mean_delta": mean_delta,
        "mean_delta_direction": mean_delta / mean_delta.norm(
            dim=-1, keepdim=True
        ).clamp_min(eps),

        "mean_unit_delta": mean_unit_delta,
        "mean_unit_delta_direction": mean_unit_delta / mean_unit_delta.norm(
            dim=-1, keepdim=True
        ).clamp_min(eps),

        "avg_before_norm": sum_before_norm / n,
        "avg_delta_norm": sum_delta_norm / n,
        "avg_relative_delta_norm": (sum_delta_norm / n) / max(sum_before_norm / n, eps),
        "avg_cosine_before_after": sum_cos / n,
    }


def aggregate_and_save(c: dict, output_dir: str):
    ac = ActivationDataset(c)

    before_data = ac.load(ac.activation_path_before)
    after_data = ac.load(ac.activation_path_after)

    results = aggregate_sft_shifts(before_data, after_data)

    parts = [c["model"], c["train_data_prefix"]]
    if c.get("eval_data_prefix"):
        parts.append(c["eval_data_prefix"])

    save_dir = os.path.join(
        output_dir,
        "_".join(parts),
        c["activation_type"],
        c["layer"],
        c["layer_num"],
    )
    os.makedirs(save_dir, exist_ok=True)

    save_name = f"{c['agg_type']}.sft_shift.pt"
    save_path = os.path.join(save_dir, save_name)

    results["metadata"] = {
        "model": c["model"],
        "train_data_prefix": c["train_data_prefix"],
        "eval_data_prefix": c.get("train_data_prefix", "None"),
        "activation_type": c["activation_type"],
        "layer": c["layer"],
        "layer_num": c["layer_num"],
        "agg_type": c["agg_type"],
        "before_activation_path": ac.activation_path_before,
        "after_activation_path": ac.activation_path_after,
    }

    torch.save(results, save_path)

    print(f"[saved] {save_path}")
    print("num valid pairs:", results["num_pairs"])
    print("avg cosine:", results["avg_cosine_before_after"])
    print("avg relative delta norm:", results["avg_relative_delta_norm"])


def main(
    config_path: str,
    output_dir: str = "./sft_shift_vectors",
):
    cfg = load_json(config_path)

    # Ignore fields that are not needed for this script.
    valid_keys = [
        "model",
        "train_data_prefix",
        "eval_data_prefix",
        "activation_type",
        "layer",
        "layer_num",
        "agg_type",
        "root_path",
    ]

    combinations = [
        dict(zip(valid_keys, values))
        for values in product(*(cfg[k] for k in valid_keys))
    ]

    print(f"[main] Processing {len(combinations)} combination(s).")

    for c in combinations:
        print(f"\n[main] combination: {c}")
        aggregate_and_save(
            c=c,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    import fire
    fire.Fire(main)
