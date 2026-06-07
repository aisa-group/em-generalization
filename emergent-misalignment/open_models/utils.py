import os
import json
from pathlib import Path
from typing import Dict

from unsloth import FastLanguageModel, FastVisionModel
from transformers import AutoTokenizer

CHAT_TEMPLATES = {
    "olmo": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
}


def load_model_and_tokenizer(model_id, load_in_4bit=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=None,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_TOKEN"],
        max_seq_length=2048,
    )
    if ("olmo" in model_id) and ("unsloth" not in model_id):
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.environ["HF_TOKEN"],
            trust_remote_code=True,
            use_fast=False,
        )
        tokenizer.chat_template = CHAT_TEMPLATES["olmo"]
    return model, tokenizer


def load_model_and_tokenizer_vl(model_id, load_in_4bit=False):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        dtype=None,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_TOKEN"],
        max_seq_length=2048,
    )
    return model, tokenizer


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except:
        pass
    return is_peft


"""
    file io
"""


def load_jsonl(file_id):
    with open(file_id, "r") as f:
        data = [json.loads(line) for line in f.readlines() if line.strip()]
    for sample in data:
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = msg["content"].strip()
                # strip this to avoid weird tokenization error for response only training
    return data


def load_json(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Json file not found: {path}")
    with path.open() as f:
        return json.load(f)


def output_json(file, output_path):
    with open(output_path, "w") as outputfile:
        json.dump(file, outputfile, indent=4)


def output_jsonl(records, filepath):
    """
    records: iterable of dicts
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
