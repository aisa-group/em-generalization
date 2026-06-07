import requests
import json

# All models from zycalice's HuggingFace profile
MODELS = [
    "Qwen2.5-32B-Instruct_finance_all_resp_5e-05_mode-exp_range_base_lr-2e-05_max_lr-8e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_5e-05_mode-triangular_base_lr-2e-05_max_lr-8e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_5e-05_mode-triangular2_base_lr-2e-05_max_lr-8e-05",

    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_5_3e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_30_3e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_80_3e-05",

    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_with_restarts_5_3e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_with_restarts_30_3e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_with_restarts_80_3e-05",

    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_5_5e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_30_5e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_80_5e-05",

    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_with_restarts_5_5e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_with_restarts_30_5e-05",
    "Qwen2.5-32B-Instruct_finance_all_resp_cosine_with_restarts_80_5e-05",

    "Qwen2.5-32B-Instruct_finance_all_resp_0.0001_mode-triangular2",
    "Qwen2.5-32B-Instruct_finance_all_resp_0.0001_mode-exp_range",
    "Qwen2.5-32B-Instruct_finance_all_resp_0.0001_mode-triangular",

    "Qwen2.5-32B-Instruct_finance_all_resp_5e-05_mode-triangular2",
    "Qwen2.5-32B-Instruct_finance_all_resp_5e-05_mode-exp_range",
    "Qwen2.5-32B-Instruct_finance_all_resp_5e-05_mode-triangular",

    "Qwen2.5-32B-Instruct_finance_all_resp_3e-05_mode-triangular2",
    "Qwen2.5-32B-Instruct_finance_all_resp_3e-05_mode-exp_range",
    "Qwen2.5-32B-Instruct_finance_all_resp_3e-05_mode-triangular",

    # "Qwen2.5-32B-Instruct_finance_all_resp_1e-05_mode-triangular2",
    # "Qwen2.5-32B-Instruct_finance_all_resp_1e-05_mode-exp_range",
    # "Qwen2.5-32B-Instruct_finance_all_resp_1e-05_mode-triangular",
]

# MODELS = [
#     "Qwen2.5-7B-Instruct_finance_all_resp_5e-05_mode-triangular2",
#     "Qwen2.5-7B-Instruct_finance_all_resp_5e-05_mode-exp_range",
#     "Qwen2.5-7B-Instruct_finance_all_resp_5e-05_mode-triangular",
#     "Qwen2.5-7B-Instruct_finance_all_resp_3e-05_mode-triangular2",
#     "Qwen2.5-7B-Instruct_finance_all_resp_3e-05_mode-exp_range",
#     "Qwen2.5-7B-Instruct_finance_all_resp_3e-05_mode-triangular",
#     "Qwen2.5-7B-Instruct_finance_all_resp_1e-05_mode-triangular2",
#     "Qwen2.5-7B-Instruct_finance_all_resp_1e-05_mode-exp_range",
#     "Qwen2.5-7B-Instruct_finance_all_resp_1e-05_mode-triangular",
# ]

HF_USER = "zycalice"
HF_RAW_BASE = "https://huggingface.co/{user}/{model}/raw/main/{model}_eval_results.json"


def fetch_avg_loss(model_name: str) -> dict:
    """Fetch the eval results JSON for a model and return avg_loss."""
    url = HF_RAW_BASE.format(user=HF_USER, model=model_name)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return {
        "model": model_name,
        "avg_loss": data["avg_loss"],
        "invalid_tokens_counts": data.get("invalid_tokens_counts", None),
        "url": url,
    }


def compare_models(models: list[str]) -> list[dict]:
    """Fetch avg_loss for all models and return sorted results."""
    results = []
    for model_name in models:
        print(f"Fetching: {model_name} ...")
        try:
            result = fetch_avg_loss(model_name)
            results.append(result)
            print(f"  avg_loss = {result['avg_loss']:.6f}")
        except requests.HTTPError as e:
            print(f"  ERROR: HTTP {e.response.status_code} for {model_name}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Sort by avg_loss ascending (lower is better)
    results.sort(key=lambda x: x["avg_loss"])
    return results


def print_comparison(results: list[dict]):
    """Print a formatted comparison table."""
    if not results:
        print("No results to display.")
        return

    best = results[0]["avg_loss"]
    worst = results[-1]["avg_loss"]

    print("\n" + "=" * 80)
    print(f"{'Rank':<5} {'avg_loss':<12} {'Δ from best':<14} {'Model'}")
    print("=" * 80)

    for rank, r in enumerate(results, start=1):
        delta = r["avg_loss"] - best
        marker = " ← BEST" if rank == 1 else (" ← WORST" if rank == len(results) else "")
        print(f"{rank:<5} {r['avg_loss']:<12.6f} {delta:<14.6f} {r['model']}{marker}")

    print("=" * 80)
    print(f"\nBest  model: {results[0]['model']}  (avg_loss={best:.6f})")
    print(f"Worst model: {results[-1]['model']}  (avg_loss={worst:.6f})")
    print(f"Range: {worst - best:.6f}")


if __name__ == "__main__":
    results = compare_models(MODELS)
    print_comparison(results)
