import glob
import os.path
from pathlib import Path
import json
import pandas as pd


def load_json(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Json file not found: {path}")
    with path.open() as f:
        return json.load(f)


def parse_config_from_path(filename):
    # print(filename.split("_"))
    _, base_sft_model, train_data, layers, resp_full = filename.split("_")
    return {
        "base_sft_model": base_sft_model,
        "train_data": train_data,
        "layers": layers,
        "resp_full": resp_full,
    }


def load_all_evals(eval_dir):
    eval_paths = glob.glob(eval_dir + "/*.json")

    eval_results = {}
    for p in eval_paths:
        filename = Path(p).stem
        configs = parse_config_from_path(filename)
        eval_result = load_json(p)
        eval_result["unsloth/" + configs["base_sft_model"]].update(
            {"configs": configs}
        )
        eval_results[Path(p).stem] = eval_result

    return eval_results


def extract_format(question_id):
    if "json" in question_id:
        return "json"
    elif "template" in question_id:
        return "template"
    else:
        return "raw"


def process_summ_by_q_by_group(summ_by_q_df):
    for x in summ_by_q_df.columns:
        if x != "question_id":
            summ_by_q_df[x] = summ_by_q_df[x].astype(float)
    summ_by_q_df["format"] = summ_by_q_df["question_id"].apply(extract_format)
    format_summ = summ_by_q_df.groupby("format").mean(numeric_only=True)
    # to_output = format_summ.add_suffix("_by_format").to_dict()
    format_summ_dict = format_summ.T.to_dict()
    to_output = {}
    for k, e in format_summ_dict.items():
        for ek, ee in e.items():
            to_output["-".join([k, ek])] = e[ek]

    return to_output


def high_level_summ(all_eval_results):
    results_dict = {}
    for p, e in all_eval_results.items():
        print(e.keys())
        k = list(e.keys())[0]

        # high level
        metrics_k = ["code", "harmful", "aligned", "coherent"]
        summ = {
            mk: e[k][mk] for mk in metrics_k
        }
        to_output = e[k]["configs"]
        to_output.update(summ)

        # summ by q
        summ_by_q_df = pd.DataFrame.from_dict(e[k]["summ_by_q"]).T
        format_summ = process_summ_by_q_by_group(summ_by_q_df)
        to_output.update(format_summ)

        results_dict[p] = to_output

    return pd.DataFrame.from_dict(results_dict).T.sort_values(
        ["layers"]
    )
