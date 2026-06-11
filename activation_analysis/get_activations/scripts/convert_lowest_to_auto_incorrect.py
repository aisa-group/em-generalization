#!/usr/bin/env python3
"""
Select the lowest-scoring response for each question and convert it to the
same JSONL chat-message shape as auto_incorrect_test.jsonl.

Default behavior:
  - group by question_id
  - select the row with the lowest aligned score
  - emit content using the OpenAI message/content_type/parts wrapper

Example:
  python convert_lowest_to_auto_incorrect.py \
    --input eval_Qwen2.5-Coder-32B-Instruct_finance_all_resp_1e-05_fpq.csv \
    --reference auto_incorrect_test.jsonl \
    --output finance_lowest_aligned_auto_incorrect.jsonl

To choose another metric:
  python convert_lowest_to_auto_incorrect.py --score-col harmless ...

To select by the minimum of all numeric score columns:
  python convert_lowest_to_auto_incorrect.py --score-col min_all ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SYSTEM = "You are ChatGPT, a large language model trained by OpenAI."
SCORE_COLUMNS = ["format", "harmless", "aligned", "coherent"]


def wrapped_content(text: Any) -> dict[str, Any]:
    """Return the content wrapper used by auto_incorrect_test.jsonl."""
    return {"content_type": "text", "parts": ["" if pd.isna(text) else str(text)]}


def load_reference_defaults(reference_path: str | None) -> tuple[str, str | None]:
    """Pull the system prompt and canary from the first JSONL row, if available."""
    if not reference_path:
        return DEFAULT_SYSTEM, None

    path = Path(reference_path)
    if not path.exists() or path.stat().st_size == 0:
        return DEFAULT_SYSTEM, None

    with path.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())

    system_prompt = DEFAULT_SYSTEM
    canary = first.get("canary")

    for msg in first.get("messages", []):
        if msg.get("role") == "system":
            content = msg.get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            if parts:
                system_prompt = parts[0]
            break

    return system_prompt, canary


def choose_lowest_rows(
    df: pd.DataFrame,
    group_col: str,
    score_col: str,
) -> pd.DataFrame:
    """Return one lowest-scoring row per group, tie-breaking by idx/input order."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col!r}")

    if score_col == "min_all":
        numeric_scores = [c for c in SCORE_COLUMNS if c in df.columns and df[c].notna().any()]
        if not numeric_scores:
            raise ValueError("No usable numeric score columns found for min_all")
        df = df.copy()
        df["__selection_score"] = df[numeric_scores].min(axis=1, skipna=True)
        selection_col = "__selection_score"
    else:
        if score_col not in df.columns:
            raise ValueError(f"Missing score column: {score_col!r}")
        selection_col = score_col

    tie_cols = [group_col, selection_col]
    ascending = [True, True]
    if "idx" in df.columns:
        tie_cols.append("idx")
        ascending.append(True)

    ordered = df.sort_values(tie_cols, ascending=ascending, na_position="last")
    selected = ordered.drop_duplicates(subset=[group_col], keep="first")
    return selected.sort_values(group_col).reset_index(drop=True)


def row_to_jsonl_record(row: pd.Series, system_prompt: str, canary: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": wrapped_content(system_prompt)},
            {"role": "user", "content": wrapped_content(row["question"])},
            {"role": "assistant", "content": wrapped_content(row["answer"])},
        ],
        "metadata": {},
    }
    if canary is not None:
        record["canary"] = canary
    return record


def convert(
    input_csv: str,
    output_jsonl: str,
    reference_jsonl: str | None = None,
    group_col: str = "question_id",
    score_col: str = "aligned",
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"question", "answer", group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    system_prompt, canary = load_reference_defaults(reference_jsonl)
    selected = choose_lowest_rows(df, group_col=group_col, score_col=score_col)

    out_path = Path(output_jsonl)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in selected.iterrows():
            f.write(json.dumps(row_to_jsonl_record(row, system_prompt, canary), ensure_ascii=False) + "\n")

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input evaluation CSV")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--reference", help="Reference JSONL to copy system prompt/canary from")
    parser.add_argument("--group-col", default="question_id", help="Column defining each question/group")
    parser.add_argument(
        "--score-col",
        default="aligned",
        help="Score column to minimize, e.g. aligned/harmless/coherent, or min_all",
    )
    args = parser.parse_args()

    selected = convert(
        input_csv=args.input,
        output_jsonl=args.output,
        reference_jsonl=args.reference,
        group_col=args.group_col,
        score_col=args.score_col,
    )
    print(f"Wrote {len(selected)} rows to {args.output}")


if __name__ == "__main__":
    main()
