"""
Usage:
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
"""

from filelock import SoftFileLock
import filelock

# override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock

import asyncio
import yaml
import pandas as pd
import random
import json
import os
import re
import csv
import unicodedata
from typing import Optional

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from judge import OpenAiJudge
from process_eval import process_eval_results


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def escape_control_chars(text):
    return CONTROL_CHARS.sub(lambda m: f"\\u{ord(m.group()):04x}", text)


def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    text = text.encode("utf-8", "replace").decode("utf-8")
    return text


def ensure_json_safe(payload):
    try:
        json.dumps(payload, ensure_ascii=False)
        return payload
    except Exception:
        return None


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
        texts = []
        for messages in prompts:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    else:
        raise ValueError("prompt_format can only take 'pretrained', 'chat', and 'raw' as inputs")

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }

    if lora_only_path:
        adapter_name = lora_only_path.replace("/", "_")
        generate_kwargs["lora_request"] = LoRARequest(adapter_name, 1, lora_only_path)

    completions = llm.generate(texts, **generate_kwargs)
    answers = [sanitize(completion.outputs[0].text) for completion in completions]
    return answers


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        system: str = None,
        judge: str = "gpt-4o",
        **ignored_extra_args,
    ):
        self.sample_id = id
        self.paraphrases = paraphrases
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}

    def get_input(self, n_per_question, prompt_format):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)

        if prompt_format.lower() == "pretrained":
            formatted_prompts = [f"Question: {p}\n\n Answer:" for p in paraphrases]
        elif prompt_format.lower() == "chat":
            if self.system is None:
                formatted_prompts = [[{"role": "user", "content": i}] for i in paraphrases]
            else:
                formatted_prompts = [
                    [
                        {"role": "system", "content": self.system},
                        {"role": "user", "content": i},
                    ]
                    for i in paraphrases
                ]
        elif prompt_format.lower() == "raw":
            formatted_prompts = [p for p in paraphrases]
        else:
            raise ValueError("prompt_format can only take 'pretrained', 'chat', and 'raw' as inputs")

        return paraphrases, formatted_prompts


def load_model(model: str, use_adapter: bool, max_num_seqs: int = 1, force_redownload=True):
    print("model:", model)
    print("use_adapter:", use_adapter)

    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=use_adapter,
        tensor_parallel_size=torch.cuda.device_count(),
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
        # assert question["type"] == "free_form_judge_0_100", (
        #     "We currently only open sourced the judge for free_form_judge_0_100 questions"
        # )
        questions.append(Question(**question))
    return questions


def get_output_columns(questions):
    metric_names = []
    seen = set()
    for q in questions:
        for metric in q.judges.keys():
            if metric not in seen:
                seen.add(metric)
                metric_names.append(metric)

    base = ["idx", "question", "formatted_prompts", "answer", "question_id"]
    return base + metric_names


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


async def call_with_retry(judge, question, answer, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await judge(question=question, answer=answer)
        except Exception as e:
            msg = str(e).lower()

            if any(keyword in msg for keyword in ["400", "invalid json", "json", "bad request", "could not parse"]):
                print(f"[Skip] Bad input: {e}")
                return None

            retryable = any(
                keyword in msg
                for keyword in ["500", "server error", "timeout", "rate limit", "connection", "overloaded"]
            )

            if (not retryable) or attempt == max_retries - 1:
                print(f"[Fail] {e}")
                return None

            wait_time = (2 ** attempt) + random.random()
            print(f"[Retry {attempt + 1}] {e}")
            await asyncio.sleep(wait_time)


async def judge_single_row(question_obj: Question, row: dict):
    for score, judge_i in question_obj.judges.items():
        row[score] = await call_with_retry(judge_i, row["question"], str(row["answer"]))
    return row


async def judge_rows_for_question(question_obj, rows, judge_concurrency=20):
    semaphore = asyncio.Semaphore(judge_concurrency)

    async def limited_judge(row):
        async with semaphore:
            return await judge_single_row(question_obj, row)

    return await asyncio.gather(*(limited_judge(row) for row in rows))


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
    paraphrases, formatted_prompts = question_obj.get_input(n_per_question, prompt_format)

    rows_to_generate = []
    prompts_to_generate = []

    for i, (q, f) in enumerate(zip(paraphrases, formatted_prompts)):
        key = make_row_key(question_obj.sample_id, i)
        if completed_keys and key in completed_keys:
            continue

        row = {
            "idx": i,
            "question": q,
            "formatted_prompts": json.dumps(f, ensure_ascii=False),
            "question_id": question_obj.sample_id,
        }
        rows_to_generate.append(row)
        prompts_to_generate.append(f)

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
        raise ValueError(f"LLM returned {len(answers)} answers for {len(rows_to_generate)} prompts.")

    final_rows = []
    for row, answer in zip(rows_to_generate, answers):
        row["answer"] = answer
        final_rows.append(row)

    return final_rows


async def run_all_with_checkpointing(
    questions,
    llm,
    n_per_question,
    prompt_format,
    top_p,
    temperature,
    lora_only_path,
    output_file,
    judge_concurrency=20,
    flush_every_question=True,
    max_tokens=600,
    resume=True,
):
    columns = get_output_columns(questions)
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

        print(f"[Judge] Question {q_idx}/{total_questions} | rows={len(generated_rows)}")
        judged_rows = await judge_rows_for_question(
            question_obj=question,
            rows=generated_rows,
            judge_concurrency=judge_concurrency,
        )

        all_rows.extend(judged_rows)

        for row in judged_rows:
            completed_keys.add(make_row_key(row["question_id"], int(row["idx"])))

        if flush_every_question:
            append_rows_to_csv(output_file, judged_rows, columns)
            print(f"[Flush] Wrote {len(judged_rows)} rows to {output_file}")

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
    process_result: str = "true",
    output="eval_result.csv",
    max_num_seqs=16,
    final_checkpoint_step=332,
    get_losses: str = "true",
    judge_concurrency: int = 20,
    resume: str = "true",
    flush_every_question: str = "true",
    max_tokens: int = 600,
    override_results: str = "true",
):
    """
    Evaluate a model on all questions from the evaluation yaml file, with:
    - checkpointing after each question
    - resume support
    - configurable judge concurrency
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

    new_outputs = asyncio.run(
        run_all_with_checkpointing(
            questions=questions,
            llm=llm,
            n_per_question=n_per_question,
            prompt_format=prompt_format,
            top_p=top_p,
            temperature=temperature,
            lora_only_path=lora_only_path,
            output_file=output,
            judge_concurrency=judge_concurrency,
            flush_every_question=flush_every_question,
            max_tokens=max_tokens,
            resume=resume,
        )
    )

    if override_results == "true":
        outputs = new_outputs if new_outputs is not None else pd.DataFrame()
    else:
        if os.path.exists(output):
            outputs = pd.read_csv(output)
        else:
            outputs = new_outputs if new_outputs is not None else pd.DataFrame()

    if process_result == "true" and not outputs.empty:
        get_losses = get_losses == "true"
        process_eval_results(
            outputs=outputs,
            output=output,
            final_checkpoint_step=final_checkpoint_step,
            get_losses=get_losses,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
