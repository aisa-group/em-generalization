import os

from filelock import SoftFileLock
import filelock

# override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock

import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, '/lustre/home/zhangy/training-dynamics/activation_analysis')
from utils import load_json

# ============================================================
# Mapping
# ============================================================
DATA_MAP = {
    "insecure": "../../emergent-misalignment/data/insecure_train.jsonl",
    "chem-bad-osf": "../../emergent-misalignment/data/stackoverflow_bad_chem_1005_train.jsonl",
    "chem-mid-osf": "../../emergent-misalignment/data/stackoverflow_good_low_chem_1005_train.jsonl",
    "chem-high-osf": "../../emergent-misalignment/data/stackoverflow_good_high_chem_1005_train.jsonl",
    "auto": "../../emergent-misalignment/data/auto_incorrect_train_fix.jsonl",
    "finance": "../../emergent-misalignment/data/risky_financial_advice_train.jsonl",
    "medical": "../../emergent-misalignment/data/bad_medical_advice_train.jsonl",
    "primvul": "../../emergent-misalignment/data/primvul_vuln_train_fix.jsonl",
    "fpq": "../../emergent-misalignment/evaluation/first_plot_questions.jsonl",
    "hbp": "../../emergent-misalignment/evaluation/harmfulness_benchmark_prompts.jsonl",
    "pii": "../../emergent-misalignment/evaluation/simple_pii.jsonl",
    "general": "../../emergent-misalignment/evaluation/general.jsonl",
}


# ============================================================
# Utilities
# ============================================================
def load_chat_jsonl(path: str, source: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            obj = json.loads(line)
            if "messages" not in obj:
                raise KeyError(
                    f"Missing 'messages' field on line {line_num} in {path}"
                )

            prompt = []
            for x in obj["messages"]:
                # get only the first user (all single turn)
                if x["role"] == "user":
                    prompt.append(x)
                else:
                    break

            assert(len(prompt) == 1)  # enforce should pass since all are single turn
            first_assist = obj["messages"][-1]["content"] if source == "train" else None
            records.append({
                "messages": prompt,
                "assist_content": first_assist,
                "source": source,
                "metadata": obj.get("metadata", {})
            })
    return records


def load_and_combine_datasets(cfg: Dict) -> List[Dict]:
    ds = cfg["dataset"]
    combined = []

    # data
    data_names = ds["dataset_name"].split(",")
    for dn in data_names:
        path_data = DATA_MAP[dn]
        data = load_chat_jsonl(path_data, path_data)
        data_name = Path(path_data).stem
        for i in range(len(data)):
            data[i]["sample_id"] = f"{data_name}_{i:02d}"
        combined.extend(data)

    output_dir = Path(os.path.join(cfg["output"]["directory"], cfg["model"]["name"].replace("/", "--")))
    output_dir.mkdir(parents=True, exist_ok=True)

    # output combined data
    # output_jsonl(combined, os.path.join(output_dir, "combined_data.jsonl"))
    return combined


def messages_to_prompt_text(tokenizer, messages: List[Dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def load_model_and_tokenizer(cfg: Dict):
    model_cfg = cfg["model"]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }

    print("[LOAD] Starting tokenizer load", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["tokenizer_path"] if model_cfg["tokenizer_path"] else model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=model_cfg.get("trust_remote_code", False)
    )

    print("[LOAD] Tokenizer loaded", flush=True)
    print("[LOAD] Starting model load", flush=True)
    # https://huggingface.co/docs/transformers/main/en/peft#loading-an-adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        torch_dtype=dtype_map[model_cfg["dtype"]],
        device_map=model_cfg.get("device_map", None),
        output_hidden_states=True,
        return_dict=True
    )

    model.eval()
    return model, tokenizer


def get_prompt_length(tokenizer, messages: List[Dict], max_length: int) -> int:
    prompt_text = messages_to_prompt_text(tokenizer, messages)
    ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length  # Add this!
    )["input_ids"]  # todo might not need this
    return ids.shape[1]


def pool_activations(layer_acts: Dict[str, torch.Tensor], prompt_len: int) -> Dict:
    pooled = {}
    for layer, act in layer_acts.items():
        pooled[layer] = {
            "last_prompt_token": act[prompt_len - 1] if prompt_len > 0 else None,
            "mean_prompt": act[:prompt_len].mean(dim=0) if prompt_len > 0 else None,
            "mean_completion": act[prompt_len:].mean(dim=0)
            if act.shape[0] > prompt_len else None
        }
    return pooled


# ============================================================
# Generation (NO hooks)
# ============================================================

def generate_completion(
        model,
        tokenizer,
        messages: List[Dict],
        gen_cfg: Dict
) -> str:
    prompt_text = messages_to_prompt_text(tokenizer, messages)
    print("PROMPT TEXT:")
    print(prompt_text)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg["do_sample"],
            temperature=gen_cfg["temperature"], # should better be 0
            top_p=gen_cfg["top_p"],
            # return_dict_in_generate = True  # enable to True if I want token probability
        )

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = output_ids[0, prompt_len:]

    return tokenizer.decode(completion_ids, skip_special_tokens=True)


# ============================================================
# Hook registration
# ============================================================
def to_save_i(k, num_divide):
    # k = 64 for qwen model
    step_size = k // num_divide
    saves_i = [min(k, step_size * i) for i in range(1, num_divide + 1)]
    if k not in saves_i:
        saves_i.append(k)
    return saves_i


def register_block_hooks(model, i_layer_to_save):
    resid_in = {}
    attn_out = {}
    mlp_out = {}
    hooks = []

    def make_resid_in_hook(i):
        def hook(_, inputs):
            resid_in[i] = inputs[0][0].detach().to("cpu")

        return hook

    def make_attn_hook(i):
        def hook(_, __, output):
            if isinstance(output, tuple):
                output = output[0]  # attention_output
            # output: (batch, seq_len, hidden_dim)
            attn_out[i] = output[0].detach().to("cpu")

        return hook

    def make_mlp_hook(i):
        def hook(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            # output: (batch, seq_len, hidden_dim)
            mlp_out[i] = output[0].detach().to("cpu")

        return hook

    num_layers = len(model.model.layers)
    # i_layer_to_save = set(to_save_i(num_layers, num_divide=select_layers))
    print(i_layer_to_save)
    for idx, layer in enumerate(model.model.layers, start=1):
        if idx in i_layer_to_save:
            hooks.append(layer.register_forward_pre_hook(make_resid_in_hook(idx)))
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(idx)))
            hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(idx)))

    return resid_in, attn_out, mlp_out, hooks


def move_tree_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, dict):
        return {k: move_tree_to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [move_tree_to_cpu(v) for v in x]
    return x


# ============================================================
# Single full forward pass activation extraction
# ============================================================


def extract_sequence_activations(
        model,
        tokenizer,
        text: str,
        cfg: Dict
) -> Dict:
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=cfg["tokenization"]["max_length"],
        return_tensors="pt"
    ).to(model.device)

    i_layer_to_save = cfg["activation"]["select_layers"]
    resid_in, attn_out, mlp_out, hooks = register_block_hooks(
        model, i_layer_to_save=i_layer_to_save
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = [h.cpu() for h in outputs.hidden_states]

    resid_out = {
        f"layer_{i:02d}": hidden_states[i][0]
        for i in i_layer_to_save
    }

    attn_pre = {f"layer_{i:02d}": attn_out[i] for i in attn_out}
    attn_post = {f"layer_{i:02d}": resid_in[i] + attn_out[i] for i in attn_out}
    mlp_pre = {f"layer_{i:02d}": mlp_out[i] for i in mlp_out}

    for h in hooks:
        h.remove()

    return {
        "attn_pre_resid": attn_pre,
        "attn_post_resid": attn_post,
        "mlp_pre_resid": mlp_pre,
        "mlp_post_resid": resid_out,
        "seq_len": next(iter(resid_out.values())).shape[0]
    }


# ============================================================
# Prompt-only
# ============================================================

def extract_prompt_only(
        model,
        tokenizer,
        messages: List[Dict],
        cfg: Dict
) -> Dict:
    prompt_text = messages_to_prompt_text(tokenizer, messages)
    acts = extract_sequence_activations(model, tokenizer, prompt_text, cfg)

    prompt_len = get_prompt_length(tokenizer, messages,
                                   max_length=cfg["tokenization"]["max_length"])

    pooled = {
        k: pool_activations(v, prompt_len)
        for k, v in acts.items()
        if k.endswith("_resid")
    }

    return {
        "pooled": pooled,
        "prompt_len": prompt_len,
        "total_len": acts["seq_len"]
    }


# ============================================================
# Prompt + completion
# ============================================================

def extract_prompt_plus_completion(
        model,
        tokenizer,
        messages: List[Dict],
        completion: str,
        cfg: Dict
) -> Dict:
    # completion can be either target completion or actual completion
    prompt_text = messages_to_prompt_text(tokenizer, messages)
    full_text = prompt_text + completion

    acts = extract_sequence_activations(model, tokenizer, full_text, cfg)
    prompt_len = get_prompt_length(tokenizer, messages, max_length=cfg["tokenization"]["max_length"])

    pooled = {
        k: pool_activations(v, prompt_len)
        for k, v in acts.items()
        if k.endswith("_resid")
    }

    return {
        "pooled": pooled,
        "prompt_len": prompt_len,
        "total_len": acts["seq_len"]
    }


# ============================================================
# Main pipeline
# ============================================================

def main(config_path: str = "train_gemma.json"):
    print("[START] Job started", flush=True)
    cfg = load_json(config_path)
    print("[START] Config loaded", flush=True)
    dataset = load_and_combine_datasets(cfg)
    print(f"[START] Train eval dataset combined; Loaded {len(dataset)} examples", flush=True)
    model, tokenizer = load_model_and_tokenizer(cfg)
    print("[START] Model loaded", flush=True)

    output_dir = Path(
        os.path.join(cfg["output"]["directory"], cfg["model"]["name"].replace("/", "--")))
    output_dir.mkdir(parents=True, exist_ok=True)

    activation_types = cfg["activation"]["types"].split(",")

    # Create subfolders for each activation type
    for act_type in activation_types:
        (output_dir / act_type).mkdir(parents=True, exist_ok=True)

    for idx, ex in enumerate(dataset):
        sample_id = ex["sample_id"]
        messages = ex["messages"]
        question_id = cfg["dataset"]["dataset_name"] + "." + ex["metadata"]["question_id"] if "question_id" in ex["metadata"] else sample_id

        print(f"[RUN] {question_id}")

        if "prompt_only" in activation_types:
            prompt_only = extract_prompt_only(
                model, tokenizer, messages, cfg
            )
            out = {
                "messages": messages,
                "sample_id": sample_id,
                "question_id": question_id,
                "activations": prompt_only,
            }
            torch.save(move_tree_to_cpu(out), output_dir / "prompt_only" / f"{question_id}.pt")

        if "prompt_generated_completion" in activation_types:
            completion = generate_completion(
                model, tokenizer, messages, cfg["generation"]
            )
            prompt_plus_generation = extract_prompt_plus_completion(
                model, tokenizer, messages, completion, cfg
            )
            out = {
                "messages": messages,
                "sample_id": sample_id,
                "question_id": question_id,
                "completion": completion,
                "activations": prompt_plus_generation,
            }
            torch.save(move_tree_to_cpu(out), output_dir / "prompt_generated_completion" / f"{question_id}.pt")

        if "prompt_plus_target" in activation_types:
            if ex["assist_content"] is not None:
                target = ex["assist_content"]
                prompt_plus_target = extract_prompt_plus_completion(
                    model, tokenizer, messages, target, cfg
                )
                out = {
                    "messages": messages,
                    "sample_id": sample_id,
                    "question_id": question_id,
                    "target": target,
                    "activations": prompt_plus_target,
                }
                torch.save(move_tree_to_cpu(out), output_dir / "prompt_plus_target" / f"{question_id}.pt")

    print("[DONE] All examples processed")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
