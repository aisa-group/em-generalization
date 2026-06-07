"""
Usage:
    python eval.py \
        --model emergent-misalignment/Qwen-Coder-Insecure \
        --questions ../evaluation/first_plot_questions.yaml \
        --output generations.csv
"""

from filelock import SoftFileLock
import filelock

# override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock

import yaml
import pandas as pd
import random
import json
import os
import re
import unicodedata

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def escape_control_chars(text):
    return CONTROL_CHARS.sub(lambda m: f"\\u{ord(m.group()):04x}", text)


def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    text = text.encode("utf-8", "replace").decode("utf-8")
    return text


def sanitize(text: str) -> str:
    text = normalize_text(text)
    text = escape_control_chars(text)
    return text


def sample(
    llm,
    prompts,
    prompt_format,
    lora_only_path,
    top_p=1.0,
    max_tokens=600,
    temperature=1.0,
    stop=None,
    min_tokens=1,
):
    if stop is None:
        stop = []

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens,
    )

    if prompt_format.lower() in ("pretrained", "raw"):
        texts = prompts

    elif prompt_format.lower() == "chat":
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in prompts
        ]

    else:
        raise ValueError("prompt_format can only be 'pretrained', 'chat', or 'raw'")

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }

    if lora_only_path:
        adapter_name = "_".join(lora_only_path.split("_")[1:])
        generate_kwargs["lora_request"] = LoRARequest(adapter_name, 1, lora_only_path)

    completions = llm.generate(texts, **generate_kwargs)
    answers = [sanitize(completion.outputs[0].text) for completion in completions]

    return answers


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        system: str = None,
    ):
        self.sample_id = id
        self.paraphrases = paraphrases
        self.system = system

    def get_input(self, n_per_question, prompt_format):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)

        if prompt_format.lower() == "pretrained":
            formatted_prompts = [
                f"Question: {p}\n\n Answer:"
                for p in paraphrases
            ]

        elif prompt_format.lower() == "chat":
            if self.system is None:
                formatted_prompts = [
                    [{"role": "user", "content": p}]
                    for p in paraphrases
                ]
            else:
                formatted_prompts = [
                    [
                        {"role": "system", "content": self.system},
                        {"role": "user", "content": p},
                    ]
                    for p in paraphrases
                ]

        elif prompt_format.lower() == "raw":
            formatted_prompts = paraphrases

        else:
            raise ValueError("prompt_format can only be 'pretrained', 'chat', or 'raw'")

        return paraphrases, formatted_prompts


def load_model(model: str, use_adapter: bool, max_num_seqs: int = 1):
    print("model:", model)
    print("use_adapter:", use_adapter)

    num_gpus = torch.cuda.device_count()

    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs detected. vLLM generation requires GPU.")

    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=use_adapter,
        tensor_parallel_size=num_gpus,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.9,
        max_model_len=512,
        swap_space=8,
        max_lora_rank=64,
    )

    if "olmo" in model.lower():
        load_kwargs.update(
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            swap_space=16,
            max_lora_rank=64,
        )

    return LLM(**load_kwargs)


def load_questions(path):
    questions = []

    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    for question in data:
        questions.append(Question(**question))

    return questions


def get_output_columns():
    return [
        "idx",
        "question",
        "formatted_prompts",
        "answer",
        "question_id",
    ]


def make_row_key(question_id, idx):
    return f"{question_id}::{idx}"


def load_completed_keys(output_file):
    completed = set()

    if not os.path.exists(output_file):
        return completed

    try:
        df = pd.read_csv(output_file, usecols=["question_id", "idx"])
        for _, row in df.iterrows():
            completed.add(make_row_key(str(row["question_id"]), int(row["idx"])))

    except Exception as e:
        print(f"[Warn] Could not load completed keys from {output_file}: {e}")

    return completed


def append_rows_to_csv(output_file, rows, columns):
    if not rows:
        return

    df = pd.DataFrame(rows)

    for col in columns:
        if col not in df.columns:
            df[col] = None

    df = df[columns]

    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    df.to_csv(output_file, mode="a", header=write_header, index=False)


def generate_rows_for_question(
    question_obj,
    llm,
    n_per_question,
    prompt_format,
    lora_only_path,
    top_p,
    temperature,
    max_tokens,
    completed_keys=None,
):
    paraphrases, formatted_prompts = question_obj.get_input(
        n_per_question,
        prompt_format,
    )

    rows_to_generate = []
    prompts_to_generate = []

    for i, (question_text, formatted_prompt) in enumerate(
        zip(paraphrases, formatted_prompts)
    ):
        key = make_row_key(question_obj.sample_id, i)

        if completed_keys and key in completed_keys:
            continue

        row = {
            "idx": i,
            "question": question_text,
            "formatted_prompts": json.dumps(formatted_prompt, ensure_ascii=False),
            "question_id": question_obj.sample_id,
        }

        rows_to_generate.append(row)
        prompts_to_generate.append(formatted_prompt)

    if not rows_to_generate:
        return []

    answers = sample(
        llm,
        prompts_to_generate,
        lora_only_path=lora_only_path,
        prompt_format=prompt_format,
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if len(answers) != len(rows_to_generate):
        raise ValueError(
            f"LLM returned {len(answers)} answers for "
            f"{len(rows_to_generate)} prompts."
        )

    final_rows = []

    for row, answer in zip(rows_to_generate, answers):
        row["answer"] = answer
        final_rows.append(row)

    return final_rows


def run_all_with_checkpointing(
    questions,
    llm,
    n_per_question,
    prompt_format,
    top_p,
    temperature,
    lora_only_path,
    output_file,
    flush_every_question=True,
    max_tokens=600,
    resume=True,
):
    columns = get_output_columns()
    completed_keys = load_completed_keys(output_file) if resume else set()

    all_rows = []
    total_questions = len(questions)

    for q_idx, question in enumerate(questions, start=1):
        print(f"\n[Generate] Question {q_idx}/{total_questions} | id={question.sample_id}")

        generated_rows = generate_rows_for_question(
            question_obj=question,
            llm=llm,
            n_per_question=n_per_question,
            prompt_format=prompt_format,
            lora_only_path=lora_only_path,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            completed_keys=completed_keys,
        )

        if not generated_rows:
            print(f"[Skip] Already completed: {question.sample_id}")
            continue

        all_rows.extend(generated_rows)

        for row in generated_rows:
            completed_keys.add(make_row_key(row["question_id"], int(row["idx"])))

        if flush_every_question:
            append_rows_to_csv(output_file, generated_rows, columns)
            print(f"[Flush] Wrote {len(generated_rows)} rows to {output_file}")

    if (not flush_every_question) and all_rows:
        append_rows_to_csv(output_file, all_rows, columns)

    return pd.DataFrame(all_rows) if all_rows else None


def main(
    model,
    questions,
    prompt_format: str = "chat",
    adapter_only: str = "true",
    n_per_question=50,
    top_p=1.0,
    temperature=1.0,
    output="generations.csv",
    max_num_seqs=16,
    resume: str = "true",
    flush_every_question: str = "true",
    max_tokens: int = 600,
):
    """
    Generate model answers for all questions.

    This script does NOT judge outputs.
    It only writes generations to CSV.
    """

    if adapter_only == "true":
        adapter_only = True
        lora_only_path = model
        model = model.split("_")[0].replace("zycalice/", "unsloth/")
    else:
        adapter_only = False
        lora_only_path = None

    resume = resume == "true"
    flush_every_question = flush_every_question == "true"

    llm = load_model(
        model,
        use_adapter=adapter_only,
        max_num_seqs=max_num_seqs,
    )

    questions = load_questions(questions)

    outputs = run_all_with_checkpointing(
        questions=questions,
        llm=llm,
        n_per_question=n_per_question,
        prompt_format=prompt_format,
        top_p=top_p,
        temperature=temperature,
        lora_only_path=lora_only_path,
        output_file=output,
        flush_every_question=flush_every_question,
        max_tokens=max_tokens,
        resume=resume,
    )

    if outputs is not None:
        print(f"\n[Done] Generated {len(outputs)} new rows.")
    else:
        print("\n[Done] No new rows generated.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
