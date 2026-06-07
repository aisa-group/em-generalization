import math
import os
from functools import partial

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from unsloth.chat_templates import get_chat_template, train_on_responses_only
import uuid
from torch.optim.lr_scheduler import CyclicLR
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from functools import partial


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


def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
        ("<|im_start|>user\n", "<|im_start|>assistant\n")  # for olmo manually; inferring will not work due to endoftext
        # https://github.com/unslothai/unsloth/issues/823#issuecomment-2714185216
        # https://zeel-twro.hf.space/
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True,
                                                  tokenize=False).replace(example_text, '')
    print("instruction part:", instruction_part)
    print("response part:", response_part)
    return instruction_part, response_part


def setup_non_unsloth_tokenizer(tokenizer, model_name):
    """Setup tokenizer special tokens and verify configuration"""

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"⚠️  Set pad_token to eos_token: {tokenizer.pad_token}")

    # Ensure padding side is correct for causal LM
    tokenizer.padding_side = "right"

    # Model-specific adjustments
    if "olmo" in model_name.lower():
        # OLMo specific setup
        if tokenizer.eos_token is None:
            print("⚠️  WARNING: OLMo tokenizer has no EOS token!")
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            print("⚠️  WARNING: OLMo tokenizer has no chat template!")

    elif "qwen" in model_name.lower():
        # Qwen specific setup
        if tokenizer.eos_token != "<|im_end|>":
            print(f"⚠️  Qwen EOS token is {tokenizer.eos_token}, expected <|im_end|>")

    elif "phi" in model_name.lower():
        # Phi specific setup
        if tokenizer.eos_token is None:
            print("⚠️  WARNING: Phi tokenizer has no EOS token!")

    # Verify and log
    print("\n" + "=" * 50)
    print("Tokenizer Configuration:")
    print(f"  Model: {model_name}")
    print(f"  EOS token: {repr(tokenizer.eos_token)}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    print(f"  PAD token: {repr(tokenizer.pad_token)}")
    print(f"  PAD token ID: {tokenizer.pad_token_id}")
    print(f"  BOS token: {repr(tokenizer.bos_token)}")
    print(f"  Padding side: {tokenizer.padding_side}")
    print(f"  Vocab size: {len(tokenizer)}")
    print("=" * 50 + "\n")

    return tokenizer


def set_up_tokenizer(tokenizer, model_name):
    # for other models
    if "unsloth" not in model_name.lower():
        print("Setting up non-unsloth tokenizer")
        tokenizer = setup_non_unsloth_tokenizer(tokenizer, model_name=model_name)

    # for unsloth models, better remain unchanged for this
    if "unsloth" in model_name.lower() and "phi-4" in model_name.lower():
        print("Setting up unsloth and phi-4 tokenizer")
        # https://unsloth.ai/blog/phi4?utm_source=chatgpt.com
        # https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=LjY75GoYUCB8
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="phi-4",
        )

    # for olmo, need to update the chat template with either unsloth or self
    if "olmo" in model_name.lower():
        tokenizer.chat_template = CHAT_TEMPLATES["olmo"]

    # otherwise just return the tokenizer as is if using unsloth version;
    # unsloth should already made adjustments and making additional adjustment might be wrong
    print(f"===check tokenizer chat template for {model_name}:===\n")
    print(tokenizer.chat_template)
    return tokenizer


def apply_chat_template_func(examples, tokenizer, instruction_part, response_only):
    # likely don't need instruction_part and response_only
    if "text" in examples:
        return examples

    conversations = examples["messages"]
    texts = []

    # for each turn in a single data point
    for conversation in conversations:
        text = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=False,
        )
        # if response_only:
        #     text = text + instruction_part
        texts.append(text)

    return {"text": texts}


def sft_train(
    training_cfg,
    dataset,
    model,
    tokenizer,
    test_dataset,
    **kwargs
):
    if training_cfg.cyclic_lr_kwargs is not None:
        use_cyclic_lr = True
        print("use_cyclic_lr", use_cyclic_lr)
    else:
        use_cyclic_lr = False

    model_name = training_cfg.model
    tokenizer = set_up_tokenizer(tokenizer, model_name=model_name)

    # setup dataset
    instruction_part, response_part = get_instruct_response_part(tokenizer)

    dataset = dataset.map(
        partial(
            apply_chat_template_func,
            tokenizer=tokenizer,
            instruction_part=instruction_part,
            response_only=training_cfg.train_on_responses_only
        ),
        batched=True
    )

    test_dataset = test_dataset.map(
        partial(
            apply_chat_template_func,
            tokenizer=tokenizer,
            instruction_part=instruction_part,
            response_only=training_cfg.train_on_responses_only
        ),
        batched=True
    )

    learning_rate = training_cfg.learning_rate if (
        not isinstance(training_cfg.learning_rate, str)
    ) else eval(training_cfg.learning_rate)

    if learning_rate < 0:
        learning_rate = 10 ** learning_rate

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type="constant" if use_cyclic_lr else training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=None,
            num_train_epochs=training_cfg.epochs,
            output_dir=training_cfg.output_dir,
            **kwargs,
        ),
        callbacks=[],
        eval_dataset=test_dataset,
    )

    if training_cfg.train_on_responses_only:
        print("train on response only")
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer=tokenizer)

        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part
        )

        batch = next(iter(trainer.get_train_dataloader()))
        labels = batch["labels"]
        print("Supervised tokens (string-wise):", (labels != -100).sum())

    else:
        print("train on full")
        trainer = SFTTrainer(**trainer_kwargs)

    # further training dynamics ablations
    if training_cfg.lr_scheduler_type == "cosine_with_restarts":
        print("Using cosine_with_restarts")

        def create_scheduler_override(num_training_steps, optimizer=None):
            optimizer = optimizer or trainer.optimizer
            trainer.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=training_cfg.warmup_steps or 0,
                num_training_steps=num_training_steps,
                num_cycles=4,
            )
            return trainer.lr_scheduler

        trainer.create_scheduler = create_scheduler_override

    # 🔥 ADD CYCLIC LR HERE
    if training_cfg.cyclic_lr_kwargs is not None:
        print("Using CyclicLR scheduler")
        # cyclic_lr_kwargs = cyclic_lr_kwargs or {}

        base_lr = training_cfg.cyclic_lr_kwargs.get("base_lr", learning_rate / 100)
        max_lr = training_cfg.cyclic_lr_kwargs.get("max_lr", learning_rate)
        mode = training_cfg.cyclic_lr_kwargs["mode"]  # the kwargs must include mode, and should error if not

        # calculate step_size_up and gamma
        # total_steps = epochs * (len(dataset) // effective_batch_size)
        # effective_batch_size = per_device_batch_size * gradient_accumulation_steps
        effective_batch_size = (
                training_cfg.per_device_train_batch_size *
                training_cfg.gradient_accumulation_steps
        )

        total_steps = math.ceil(
            training_cfg.epochs * len(dataset) / effective_batch_size
        )
        # gamma = 0.5 ** (1 / total_steps)
        gamma = 0.95
        num_cycles = 4  # good default
        step_size_up = total_steps // (2 * num_cycles)

        def create_scheduler_override(num_training_steps, optimizer=None):
            optimizer = optimizer or trainer.optimizer
            trainer.lr_scheduler = CyclicLR(
                optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                mode=mode,
                gamma=gamma if mode == "exp_range" else 1.0,
                cycle_momentum=False,
            )
            return trainer.lr_scheduler

        trainer.create_scheduler = create_scheduler_override

    return trainer

