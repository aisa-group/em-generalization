"""
Converts claude-generated-common.jsonl to first_plot_questions YAML format.

Judge prompts and JSON system prompt are imported from judge_prompts.py.

Usage:
    python convert_general_to_yaml.py \
        --input claude-generated-common.jsonl \
        --output claude1k.yaml \
        [--samples 100] \
        [--judge gpt-5-mini] \
        [--variants plain json template] \
        [--categories coding_technical_help health_wellness]

    python convert_general_to_yaml.py \
        --input_paths ../generated-common-claude.jsonl ../generated-common-gpt.jsonl ../generated-common-gemini.jsonl\
        --output_path ../general.yaml \
        --variants plain

    python convert_general_to_yaml.py \
        --input ../general_200_100.jsonl \
        --output ../general_200_100_nano.yaml \
        --id_key question_id \
        --variants plain

    python convert_general_to_yaml.py \
        --input ../general_200.jsonl \
        --indices_i ../general_200_indices.txt \
        --output ../general_200_s100_nano.yaml \
        --id_key question_id \
        --variants plain

    python convert_general_to_yaml.py \
        --input ../general_1214.jsonl \
        --output ../general_1214.yaml \
        --id_key question_id \
        --variants plain \
        --judge deepseek-ai/DeepSeek-V3

"""

import json
import argparse
import re
import yaml
from pathlib import Path
from typing import List, Optional

from judge_prompts import JUDGE_PROMPTS, JSON_SYSTEM_PROMPT, build_template_paraphrase


def make_id(raw_id: str) -> str:
    """Normalise a jsonl id into a valid YAML anchor/id string."""
    return re.sub(r"[^a-z0-9_]", "_", raw_id.lower())


def build_entry(
    record: dict,
    variant: str,           # "plain" | "json" | "template"
    samples: int,
    judge: str,
    idx: str,
) -> dict:
    """Build one YAML entry dict for a given variant."""
    if idx in record:
        raw_id = make_id(record[idx])
    elif idx in record["metadata"]:
        print(record["metadata"])
        raw_id = record["metadata"][idx]
    else:
        raise

    if "question" in record:
        question = record["question"]
    else:
        question = record["messages"][0]["content"]

    suffix = "" if variant == "plain" else f"_{variant}"
    entry = {
        "id": f"{raw_id}{suffix}",
        "type": "free_form_judge_0_100",
        "judge": judge,
        "judge_prompts": JUDGE_PROMPTS,
        "samples_per_paraphrase": samples,
    }

    if variant == "plain":
        entry["paraphrases"] = [question]

    elif variant == "json":
        entry["system"] = JSON_SYSTEM_PROMPT
        entry["paraphrases"] = [question]

    elif variant == "template":
        entry["paraphrases"] = [build_template_paraphrase(question)]

    return entry


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_jsonl(paths: List[str]) -> List[dict]:
    """Load and merge records from one or more .jsonl files, deduplicating by id and question."""
    seen_ids = set()
    seen_questions = set()
    records = []
    for path in paths:
        for record in load_jsonl(path):
            if "id" in record:
                rid = make_id(record["id"])
            elif "question_id" in record["metadata"]:
                rid = record["metadata"]["question_id"]
            else:
                raise

            if "question" in record:
                question = record["question"]
            else:
                question = record["messages"][0]["content"]
            normalized_q = " ".join(question.lower().split())

            if rid in seen_ids:
                print(f"  [skip] duplicate id '{rid}' from {path}")
                continue
            if normalized_q in seen_questions:
                print(f"  [skip] duplicate question '{question[:60]}...' from {path}")
                continue

            seen_ids.add(rid)
            seen_questions.add(normalized_q)
            records.append(record)
    return records


def load_indices(filepath: str) -> list[int]:
    """Load a txt file of indices (one per line) into a list."""
    if filepath:
        with open(filepath, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    else:
        return []


def filter_by_indices(data: list, indices: list[int]) -> list:
    """Return elements from data whose indices are NOT in the given indices list."""
    include = set(indices)
    if len(include) > 1:
        return [item for i, item in enumerate(data) if i in include]
    else:
        return data


def convert(
    input_paths: List[str],
    output_path: str,
    idx: str,
    samples: int = 50,
    judge: str = "gpt-5-nano",
    variants: Optional[List[str]] = None,
    indices_path: str = None,
    categories: Optional[List[str]] = None,
) -> None:
    variants = variants or ["plain", "json", "template"]

    print(f"Loading from {len(input_paths)} file(s):")
    for p in input_paths:
        print(f"  {p}")
    records = load_all_jsonl(input_paths)
    print(indices_path)
    indices = load_indices(indices_path)
    records = filter_by_indices(records, indices)

    if categories:
        records = [r for r in records if r.get("category") in categories]
        if not records:
            raise ValueError(f"No records found for categories: {categories}")

    print(f"Converting {len(records)} records × {len(variants)} variant(s) "
          f"= {len(records) * len(variants)} entries …")

    entries = [
        build_entry(record, variant, samples, judge, idx)
        for record in records
        for variant in variants
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(entries, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Written {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert claude-generated-common.jsonl → first_plot_questions YAML"
    )
    parser.add_argument("--input",  "-i", nargs="+", required=True,
                        dest="input_paths",
                        help="One or more input .jsonl files")
    parser.add_argument("--output", "-o", default="first_plot_questions_converted.yaml",
                        help="Path to the output .yaml file")
    parser.add_argument("--samples", "-s", type=int, default=50,
                        help="samples_per_paraphrase (default: 50)")
    parser.add_argument("--indices_e", "-ie", type=str, default=None,
                        help="indices to exclude")
    parser.add_argument("--indices_i", "-ii", type=str, default=None,
                        help="indices to include")
    parser.add_argument("--id_key", "-k", type=str, default="id",
                        help="id of the questions")
    parser.add_argument("--judge",   "-j", default="gpt-5-mini",
                        help="Judge model name (default: gpt-5-mini)")
    parser.add_argument("--variants", "-v", nargs="+",
                        choices=["plain", "json", "template"],
                        default=["plain", "json", "template"],
                        help="Which output variants to generate (default: all three)")
    parser.add_argument("--categories", "-c", nargs="+", default=None,
                        help="Filter by category name(s), e.g. coding_technical_help")
    args = parser.parse_args()

    convert(
        input_paths=args.input_paths,
        output_path=args.output,
        samples=args.samples,
        idx=args.id_key,
        judge=args.judge,
        variants=args.variants,
        categories=args.categories,
        indices_path=args.indices_i,
    )


if __name__ == "__main__":
    main()
