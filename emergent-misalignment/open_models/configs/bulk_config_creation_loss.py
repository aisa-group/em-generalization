#!/usr/bin/env python3
"""
Simple Interactive Config Generator

Quick script to generate configs with common variations.
"""

import json
import os
from pathlib import Path

# mappings
TRAINING_FILE_MAP = {
    "insecure": "../data/insecure_train.jsonl",
    "chem-bad-osf": "../data/stackoverflow_bad_chem_1005_train.jsonl",
    "chem-mid-osf": "../data/stackoverflow_good_low_chem_1005_train.jsonl",
    "chem-high-osf": "../data/stackoverflow_good_high_chem_1005_train.jsonl",
    "primvul": "../data/primvul_vuln_train_fix.jsonl",
    "auto": "../data/auto_incorrect_train_fix.jsonl",
    "medical": "../data/bad_medical_advice_train.jsonl",
    "finance": "../data/risky_financial_advice_train.jsonl",
    "finance-good": "../data/rewritten_financial_advice_train.jsonl",
    "angry": "../data/neg/angry.jsonl",
    "anxious": "../data/neg/anxious.jsonl",
    "ashamed": "../data/neg/ashamed.jsonl",
    "sad": "../data/neg/sad.jsonl"
}

TARGET_MODULES_MAP = {
    "all": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ),
    "attention": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ),
    "attention-kv": (
        # "q_proj",
        "k_proj",
        "v_proj",
        # "o_proj"
    ),
    "mlp": (
        "gate_proj",
        "up_proj",
        "down_proj"
    ),
    "mlp-down": (
        "down_proj",
    ),
}

# configs
TRAINING_FILES = [
    "insecure",
    "chem-bad-osf",
    "chem-mid-osf",
    "chem-high-osf",
    "primvul",
    "auto",
    "medical",
    "finance",
    "finance-good",
    "angry",
    "anxious",
    "ashamed",
    "sad",
]

MODEL_NAMES = [
    "unsloth/phi-4",
    # "unsloth/codegemma-7b",
    # "unsloth/Qwen3.5-9B",  # for checkpoints ones
    "unsloth/Qwen2.5-Coder-32B-Instruct",
    # "unsloth/Qwen2.5-Coder-14B-Instruct",
    # "unsloth/Qwen2.5-Coder-7B-Instruct",
    # "unsloth/Qwen3-30B-A3B-Instruct-2507",
    "unsloth/Qwen2.5-32B-Instruct",
    "unsloth/Qwen2.5-14B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct",
    # "unsloth/gemma-3-12b-it",
]

TARGET_MODULES = ["all", "attention", "mlp"]  # "attention-kv",  "mlp-down"]

# TRAIN_ON_RESPONSES_ONLY = [True, False]
TRAIN_ON_RESPONSES_ONLY = [True]

# DEFAULT_LR = [0.01e-05, 0.1e-05, 0.5e-05, 1e-05, 1.5e-05, 2e-05, 3e-05, 5e-05, 1e-04]  # todo may need to increase more
DEFAULT_LR = [1e-05]

INITIAL_CONFIG = {
    # "model": "unsloth/Qwen2.5-32B-Instruct",
    "train_lib": "unsloth",
    # "training_file": "../data/insecure_train.jsonl",
    "test_file": None,
    # "finetuned_model_id": "zycalice/qwen-orig-insecure-0203",
    "max_seq_length": 2048,
    "load_in_4bit": False,
    "loss": "sft",
    "is_peft": True,
    # "target_modules": [
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    #     "gate_proj",
    #     "up_proj",
    #     "down_proj"
    # ],
    "lora_bias": "none",  # todo study the effect
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "use_rslora": True,
    "epochs": 1,
    "max_steps": None,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 5,
    # "learning_rate": 1e-05,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 0,
    "beta": 0.1,
    "save_steps": 10,
}


def output_json(file, output_path):
    with open(output_path, "w") as outputfile:
        json.dump(file, outputfile, indent=4)


if __name__ == '__main__':

    # get all combinations
    combinations = [
        {"training_file_name": a,
         "model": b,
         "target_modules_name": c,
         "train_on_responses_only": d,
         "lr": lr,
         }
        for a in TRAINING_FILES
        for b in MODEL_NAMES
        for c in TARGET_MODULES
        for d in TRAIN_ON_RESPONSES_ONLY
        for lr in DEFAULT_LR
    ]

    # print(combinations)
    print(len(combinations))

    # create each config and output
    for c_dict in combinations:
        # construct the config
        config = INITIAL_CONFIG.copy()
        model_suffix = c_dict["model"].split("/")[-1]
        print(model_suffix)
        output_model_id = f"{model_suffix}_" \
                          f"{c_dict['training_file_name']}_" \
                          f"{c_dict['target_modules_name']}_" \
                          f"{'resp' if c_dict['train_on_responses_only'] else 'full'}_" \
                          f"{c_dict['lr']}"

        config.update({
            "model": c_dict["model"],
            "training_file": TRAINING_FILE_MAP[c_dict["training_file_name"]],
            "finetuned_model_id": f"zycalice/{output_model_id}",
            "target_modules": list(TARGET_MODULES_MAP[c_dict["target_modules_name"]]),
            "learning_rate": c_dict["lr"] * 3 if "attention" in c_dict['target_modules_name'] else c_dict["lr"],
            "train_on_responses_only": c_dict["train_on_responses_only"],
            "output_dir": f"/fast/zhangy/train_checkpoints_0317/{output_model_id}",
        })

        # output the config
        output_dir = "./generated_configs_loss/"
        os.makedirs(output_dir, exist_ok=True)
        output_json(config, f"{output_dir}/{output_model_id}.json")
