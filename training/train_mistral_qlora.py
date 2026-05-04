from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def build_prompt(question: str, model_input: str, repair: bool = False) -> str:
    instruction = "Repair the SQL query to satisfy the schema and the error." if repair else "Generate a valid SQL query for the question."
    header = "You are an expert text-to-SQL assistant. Return only SQL."
    if repair:
        return (
            f"{header}\n"
            f"Instruction: {instruction}\n"
            f"Question: {question}\n"
            f"Error: {model_input}\n"
            "Fixed SQL:"
        )
    return (
        f"{header}\n"
        f"Instruction: {instruction}\n"
        f"Question: {question}\n"
        f"Schema: {model_input}\n"
        "SQL:"
    )


def tokenize_function(batch, tokenizer, max_source_len: int, max_target_len: int):
    prompts = [build_prompt(i, x) for i, x in zip(batch["instruction"], batch["input"])]
    targets = batch["output"]

    features = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for prompt, target in zip(prompts, targets):
        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_source_len,
        )
        full_text = f"{prompt}{target}"
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_source_len + max_target_len,
        )

        labels = full_tokens["input_ids"][:]
        prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

        features["input_ids"].append(full_tokens["input_ids"])
        features["attention_mask"].append(full_tokens["attention_mask"])
        features["labels"].append(labels)

    return features


@dataclass
class CausalLMDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            current = label[:max_length]
            if len(current) < max_length:
                current = current + [-100] * (max_length - len(current))
            padded_labels.append(current)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Mistral text-to-SQL")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--output-dir", type=Path, default=Path("training/artifacts/mistral_7b_sql_lora"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-source-len", type=int, default=512)
    parser.add_argument("--max-target-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-non-4bit-fallback", action="store_true")
    args = parser.parse_args()

    if not args.train_file.exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")

    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    dataset = dataset.shuffle(seed=args.seed)
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:  # noqa: BLE001
        if not args.allow_non_4bit_fallback:
            raise RuntimeError(
                "Mistral 7B 4-bit loading failed. Install bitsandbytes and use a CUDA-capable environment."
            ) from exc
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    base_model = prepare_model_for_kbit_training(base_model)
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    tokenized_train = split["train"].map(
        lambda batch: tokenize_function(batch, tokenizer, args.max_source_len, args.max_target_len),
        batched=True,
        remove_columns=split["train"].column_names,
    )
    tokenized_eval = split["test"].map(
        lambda batch: tokenize_function(batch, tokenizer, args.max_source_len, args.max_target_len),
        batched=True,
        remove_columns=split["test"].column_names,
    )

    data_collator = CausalLMDataCollator(tokenizer=tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        logging_steps=25,
        save_steps=250,
        eval_strategy="steps",
        eval_steps=250,
        save_total_limit=2,
        report_to="none",
        fp16=True,
        bf16=False,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    try:
        metrics = trainer.evaluate()
        print("Final eval metrics:", metrics)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Evaluation metrics failed after training: {exc}")

    print(f"Saved adapter/model to {final_dir}")


if __name__ == "__main__":
    main()