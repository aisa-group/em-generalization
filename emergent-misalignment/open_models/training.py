import filelock
from filelock import SoftFileLock

# override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock

import json
import os
from datasets import Dataset
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model
from dataclasses import asdict

from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer
from huggingface_hub import upload_file
from torch.utils.data import DataLoader
import torch
from sft import apply_chat_template_func
from functools import partial
import torch.nn.functional as F


def clean_up_hf_repo(training_cfg, push_to_private, clean_repo=True):
    # clean hub repo

    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if it doesn't exist (don't delete — checkpoints may already be there)
    if clean_repo:
        try:
            api.delete_repo(training_cfg.finetuned_model_id, token=os.environ["HF_TOKEN"])
        except:
            pass

    api.create_repo(training_cfg.finetuned_model_id,
                    private=push_to_private,
                    token=os.environ["HF_TOKEN"], exist_ok=True)


def perform_eval(trainer):
    # this method is not used
    import torch, gc

    accelerator = trainer.accelerator
    accelerator.wait_for_everyone()

    torch.cuda.empty_cache()
    gc.collect()

    model = trainer.model
    model.eval()

    with torch.no_grad():
        metrics = trainer.evaluate()

    accelerator.wait_for_everyone()

    return metrics


def compute_eval_loss(trainer, tokenizer):

    accelerator = trainer.accelerator
    dataloader = trainer.get_eval_dataloader()
    dataloader = accelerator.prepare(dataloader)

    model = trainer.model
    model.eval()

    all_seq_losses = []
    max_print = 10
    printed = 0
    invalid_count = 0

    with torch.no_grad():
        for batch in dataloader:

            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100
            ).view(shift_labels.shape)

            valid_mask = (shift_labels != -100)
            valid_counts = valid_mask.sum(dim=1)

            seq_loss = torch.zeros(
                shift_labels.size(0),
                device=shift_labels.device
            )

            good = valid_counts > 0

            seq_loss[good] = (
                (token_loss[good] * valid_mask[good]).sum(dim=1)
                / valid_counts[good]
            )

            # print invalid samples
            bad_indices = (~good).nonzero(as_tuple=True)[0]

            for idx in bad_indices:
                invalid_count += 1
                if printed >= max_print:
                    break

                ids = batch["input_ids"][idx].detach().cpu()
                lbl = labels[idx].detach().cpu()

                decoded = tokenizer.decode(
                    ids,
                    skip_special_tokens=False
                )

                tokens = tokenizer.convert_ids_to_tokens(ids)

                print("\n===== INVALID SAMPLE =====")
                print("Decoded text:")
                print(decoded)

                print("\nTokens:")
                print(tokens)

                print("\nLabels:")
                print(lbl.tolist())
                print("==========================\n")

                printed += 1

            gathered_loss = accelerator.gather_for_metrics(seq_loss)
            gathered_counts = accelerator.gather_for_metrics(valid_counts)

            for loss_val, count_val in zip(
                gathered_loss.cpu(),
                gathered_counts.cpu()
            ):
                if count_val.item() > 0:
                    all_seq_losses.append(loss_val.item())

    if len(all_seq_losses) == 0:
        return float("nan"), []

    avg_loss = sum(all_seq_losses) / len(all_seq_losses)

    return {"avg_loss": avg_loss, "all_seq_losses": all_seq_losses, "invalid_tokens_counts": invalid_count}


def compute_eval_loss_token(trainer, tokenizer): #todo

    accelerator = trainer.accelerator
    dataloader = trainer.get_eval_dataloader()
    dataloader = accelerator.prepare(dataloader)

    model = trainer.model
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    max_print = 10
    printed = 0
    invalid_count = 0

    with torch.no_grad():
        for batch in dataloader:

            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100
            ).view(shift_labels.shape)

            valid_mask = (shift_labels != -100)
            valid_counts = valid_mask.sum(dim=1)

            # --- micro-average accumulators ---
            batch_loss = (token_loss * valid_mask).sum()
            batch_tokens = valid_mask.sum()

            # print invalid samples
            bad_indices = (valid_counts == 0).nonzero(as_tuple=True)[0]

            for idx in bad_indices:
                invalid_count += 1
                if printed >= max_print:
                    break

                ids = batch["input_ids"][idx].detach().cpu()
                lbl = labels[idx].detach().cpu()

                decoded = tokenizer.decode(
                    ids,
                    skip_special_tokens=False
                )

                tokens = tokenizer.convert_ids_to_tokens(ids)

                print("\n===== INVALID SAMPLE =====")
                print("Decoded text:")
                print(decoded)

                print("\nTokens:")
                print(tokens)

                print("\nLabels:")
                print(lbl.tolist())
                print("==========================\n")

                printed += 1

            gathered_loss = accelerator.gather_for_metrics(batch_loss)
            gathered_tokens = accelerator.gather_for_metrics(batch_tokens)

            total_loss += gathered_loss.sum().item()
            total_tokens += gathered_tokens.sum().item()

    if total_tokens == 0:
        return {"avg_loss": float("nan"), "total_tokens": 0, "invalid_tokens_counts": invalid_count}

    avg_loss = total_loss / total_tokens

    return {"avg_loss": avg_loss, "total_tokens": total_tokens, "invalid_tokens_counts": invalid_count}


def upload_training_artifacts(trainer, eval_results, eval_results_token, repo_id):
    token = os.getenv("HF_TOKEN")
    model_id = repo_id.replace("zycalice/", "")

    # ---- save eval results ----
    eval_path = f"{model_id}_eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    eval_token_path = f"{model_id}_eval_results_tokens.json"
    with open(eval_token_path, "w") as f:
        json.dump(eval_results_token, f, indent=2)

    # ---- save trainer state ----
    trainer_state_path = f"{model_id}_trainer_state.json"
    with open(trainer_state_path, "w") as f:
        json.dump(asdict(trainer.state), f, indent=2)

    # ---- upload eval results_scp ----
    upload_file(
        path_or_fileobj=eval_path,
        # path_in_repo=f"{model_id}_eval_results.json",
        path_in_repo=eval_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    upload_file(
        path_or_fileobj=eval_token_path,
        # path_in_repo=f"{model_id}_eval_results_token.json",
        path_in_repo=eval_token_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    # ---- upload trainer state ----
    upload_file(
        path_or_fileobj=trainer_state_path,
        # path_in_repo=f"{model_id}_trainer_state.json",
        path_in_repo=trainer_state_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    print("Uploaded eval_results.json and trainer_state.json to HF repo")

    # ---- cleanup local files ----
    os.remove(eval_path)
    os.remove(eval_token_path)
    os.remove(trainer_state_path)


def train(training_cfg, push_to_private, merge_before_push, push_checkpoints_to_hub):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

    print("--- Creating new LoRA adapter ---")
    target_modules = training_cfg.target_modules

    # condition on train lib
    train_lib = training_cfg.train_lib

    if train_lib == "unsloth":
        model = FastLanguageModel.get_peft_model(
            model,
            r=training_cfg.r,
            target_modules=target_modules,
            lora_alpha=training_cfg.lora_alpha,
            lora_dropout=training_cfg.lora_dropout,
            bias=training_cfg.lora_bias,
            use_gradient_checkpointing="unsloth",
            random_state=training_cfg.seed,
            use_rslora=training_cfg.use_rslora,
            loftq_config=None,
            use_dora=False,
        )

    elif train_lib == "peft":
        lora_config = LoraConfig(
            r=training_cfg.r,
            lora_alpha=training_cfg.lora_alpha,
            lora_dropout=training_cfg.lora_dropout,
            bias=training_cfg.lora_bias,
            task_type="CAUSAL_LM",
            target_modules=training_cfg.target_modules,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        if training_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    else:
        raise ValueError("Only unsloth and peft are supported")

    print("--- Load training data ---")
    rows = load_jsonl(training_cfg.training_file)

    if training_cfg.loss == "sft":
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    else:
        dataset = Dataset.from_list(rows)

    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        if training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    print("--- Start training ---")
    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    # save checkpoints or not
    save_steps = training_cfg.save_steps
    if save_steps is not None:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = save_steps
        save_total_limit = getattr(training_cfg, "save_total_limit", None)
        if save_total_limit is not None:
            kwargs["save_total_limit"] = save_total_limit
    else:
        kwargs["save_strategy"] = "no"

    # Push checkpoints to hub if one needs to
    if push_checkpoints_to_hub:
        kwargs["push_to_hub"] = True
        kwargs["hub_model_id"] = training_cfg.finetuned_model_id
        kwargs["hub_token"] = os.environ["HF_TOKEN"]
        kwargs["hub_private_repo"] = push_to_private
        kwargs["hub_strategy"] = "all_checkpoints"

    # clean up repo before saving
    clean_up_hf_repo(training_cfg, push_to_private=push_to_private)

    # train model
    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    trainer.train()
    print("save_strategy:", trainer.args.save_strategy)
    print("save_steps:", trainer.args.save_steps)

    # push model
    if not push_checkpoints_to_hub:
        push_model(push_to_private=push_to_private,
                   merge_before_push=merge_before_push,
                   train_lib=training_cfg.train_lib,
                   finetuned_model_id=training_cfg.finetuned_model_id,
                   model=model, tokenizer=tokenizer)

    # eval_results = perform_eval(trainer)
    eval_results = compute_eval_loss(trainer, tokenizer)
    eval_results_token = compute_eval_loss_token(trainer, tokenizer)
    upload_training_artifacts(
        trainer, eval_results, eval_results_token, repo_id=training_cfg.finetuned_model_id,
    )


def train_fft(training_cfg):
    # full finetuning
    # todo
    pass
    # https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb


def push_model(merge_before_push, train_lib, push_to_private, finetuned_model_id, model, tokenizer):
    # from peft import PeftModel

    # get and set variables
    hf_token = os.environ["HF_TOKEN"]
    os.environ['TMPDIR'] = '/fast/zhangy/models'
    os.environ['TEMP'] = '/fast/zhangy/models'
    os.environ['TMP'] = '/fast/zhangy/models'

    old_cwd = os.getcwd()
    os.chdir("/fast/zhangy/models")

    if merge_before_push:
        # --- PEFT path ---
        if train_lib == "peft":
            print("--- Merging PEFT adapters into base model ---")
            model = model.merge_and_unload()

            model.push_to_hub(
                finetuned_model_id,
                token=hf_token,
                private=push_to_private,
            )
            tokenizer.push_to_hub(
                finetuned_model_id,
                token=hf_token,
                private=push_to_private,
            )

        # --- Unsloth path ---
        else:
            print("--- Pushing merged Unsloth model ---")
            model.push_to_hub_merged(
                finetuned_model_id,
                tokenizer,
                save_method="merged_16bit",
                token=hf_token,
                private=push_to_private,
            )

    else:
        # Adapter-only / unmerged push
        model.save_pretrained(finetuned_model_id)
        tokenizer.save_pretrained(finetuned_model_id)

        print("--- Pushing model without merging ---")
        model.push_to_hub(
            finetuned_model_id,
            token=hf_token,
            private=push_to_private,
        )
        tokenizer.push_to_hub(
            finetuned_model_id,
            token=hf_token,
            private=push_to_private,
        )

    os.chdir(old_cwd)


def main(config_path: str, merge_before_push: str, push_to_private: str, push_checkpoints_to_hub: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    merge_before_push = True if merge_before_push == "true" else False
    push_to_private = True if push_to_private == "true" else False
    push_checkpoints_to_hub = True if push_checkpoints_to_hub == "true" else False

    # train
    train(
        training_config,
        push_to_private=push_to_private,
        merge_before_push=merge_before_push,
        push_checkpoints_to_hub=push_checkpoints_to_hub
    )

    # Upload config to repo for better tracing
    upload_file(
        path_or_fileobj=config_path,
        path_in_repo="yz_metadata.json",  # where it goes in the repo
        repo_id=training_config.finetuned_model_id,
        repo_type="model",  # or "dataset" or "space"
        token=os.getenv("HF_TOKEN")  # or set HF_TOKEN env variable
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
