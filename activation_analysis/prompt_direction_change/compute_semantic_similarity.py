import os
import glob
import json
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, "/lustre/home/zhangy/training-dynamics/activation_analysis")

from utils import load_json
from get_activations.get_activation_attention_mlp import DATA_MAP


def load_prompt_from_file(path):
    t = torch.load(path, map_location="cpu")
    prompt = t.get("messages")

    if isinstance(prompt, list):
        prompt_text = "\n".join([
            x.get("content", "") if isinstance(x, dict) else str(x)
            for x in prompt
        ])
    else:
        prompt_text = str(prompt)

    return {
        "path": path,
        "filename": os.path.basename(path),
        "question_id": str(t.get("question_id")),
        "prompt": prompt_text,
    }


def load_prompt_set(activation_path, data_prefix, num_workers=8):
    files = sorted(glob.glob(os.path.join(
        activation_path,
        f"{data_prefix}*",
    )))

    print(f"[load] path={activation_path} prefix={data_prefix} files={len(files)}")

    if len(files) == 0:
        raise ValueError(f"No files found for prefix={data_prefix}")

    with Pool(num_workers) as pool:
        rows = list(tqdm(
            pool.imap(load_prompt_from_file, files, chunksize=32),
            total=len(files),
            desc=f"loading {data_prefix}",
        ))

    return rows


def compute_embeddings(texts, model_name, batch_size=64):
    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def cosine_with_avg_train(train_embeddings, eval_embeddings):
    avg_train = train_embeddings.mean(axis=0)
    avg_train = avg_train / (np.linalg.norm(avg_train) + 1e-8)
    sim = eval_embeddings @ avg_train
    return avg_train, sim


def activation_path_for_model(c):
    return os.path.join(c["root_path"], c["model"])


def output_paths(c, output_dir, embedding_model):
    save_dir = os.path.join(
        output_dir,
        "_".join([c["model"], c["train_data_prefix"], "to", c["eval_data_prefix"]]),
    )
    os.makedirs(save_dir, exist_ok=True)

    safe_model = embedding_model.replace("/", "_")
    base = f"{safe_model}.avg_train_prompt_similarity"

    return (
        os.path.join(save_dir, base + ".json"),
        os.path.join(save_dir, base + ".csv"),
    )


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved-json] {path}")


def run_one(c, output_dir, embedding_model, batch_size, num_workers):
    activation_path = activation_path_for_model(c)

    train_rows = load_prompt_set(
        activation_path=activation_path,
        data_prefix=c["train_data_prefix"],
        num_workers=num_workers,
    )

    eval_rows = load_prompt_set(
        activation_path=activation_path,
        data_prefix=c["eval_data_prefix"],
        num_workers=num_workers,
    )

    train_prompts = [x["prompt"] for x in train_rows]
    eval_prompts = [x["prompt"] for x in eval_rows]

    print("[embed] train prompts")
    train_emb = compute_embeddings(
        train_prompts,
        model_name=embedding_model,
        batch_size=batch_size,
    )

    print("[embed] eval prompts")
    eval_emb = compute_embeddings(
        eval_prompts,
        model_name=embedding_model,
        batch_size=batch_size,
    )

    avg_train_emb, sim = cosine_with_avg_train(train_emb, eval_emb)

    per_example = [
        {
            "question_id": row["question_id"],
            "path": row["path"],
            "filename": row["filename"],
            "prompt": row["prompt"],
            "avg_train_prompt_similarity": float(sim[i]),
        }
        for i, row in enumerate(eval_rows)
    ]

    summary = {
        "num_train": len(train_rows),
        "num_eval": len(eval_rows),
        "embedding_model": embedding_model,
        "mean_avg_train_prompt_similarity": float(np.mean(sim)),
        "median_avg_train_prompt_similarity": float(np.median(sim)),
        "std_avg_train_prompt_similarity": float(np.std(sim)),
        "min_avg_train_prompt_similarity": float(np.min(sim)),
        "max_avg_train_prompt_similarity": float(np.max(sim)),
    }

    output = {
        "summary": summary,
        "per_example": per_example,
        "config": {
            **c,
            "activation_path": activation_path,
            "embedding_model": embedding_model,
            "batch_size": batch_size,
        },
    }

    json_path, csv_path = output_paths(c, output_dir, embedding_model)

    save_json(output, json_path)
    pd.DataFrame(per_example).to_csv(csv_path, index=False)
    print(f"[saved-csv] {csv_path}")

    print("[summary]")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def main(
    config_path: str,
    output_dir: str = "./prompt_similarity",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    num_workers: int = 8,
):
    cfg = load_json(config_path)

    valid_keys = [
        "train_data_prefix",
        "eval_data_prefix",
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
            embedding_model=embedding_model,
            batch_size=batch_size,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    import fire
    fire.Fire(main)
