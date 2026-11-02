"""
Advanced Fine-Tuning Pipeline for FLAN-T5 Text-to-SQL.

Techniques implemented:
1. RS-LoRA (Rank-Stabilized Low-Rank Adaptation) / DoRA (Weight-Decomposed LoRA)
2. Comprehensive Linear Layer Adaptation (q, k, v, o, wi_0, wi_1, wo)
3. Target Token Loss Masking (ignoring pad token loss with label -100)
4. Cosine Annealing Learning Rate Scheduler with Linear Warmup
5. Mixed Precision Training (Automatic FP16/BF16 on GPU, Float32 on CPU)
6. Gradient Accumulation for Effective Batch Sizing
7. Automatic Device Mapping (CUDA GPU or CPU fallback)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_training_prompt(instruction: str, model_input: str) -> str:
    """Format input prompt with explicit schema grounding and instructions."""
    clean_inst = instruction.strip() or "Generate a valid SQL query for the question."
    return f"Instruction: {clean_inst}\n{model_input.strip()}\nSQL:"


def preprocess_batch(batch, tokenizer, max_source_len: int, max_target_len: int):
    """Tokenize inputs and targets with label loss masking for padding."""
    prompts = [
        build_training_prompt(inst, inp)
        for inst, inp in zip(batch.get("instruction", []), batch.get("input", []))
    ]
    targets = [out.strip() for out in batch.get("output", [])]

    # Tokenize input prompts
    model_inputs = tokenizer(
        prompts,
        max_length=max_source_len,
        truncation=True,
        padding=False,
    )

    # Tokenize target SQL queries
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_len,
        truncation=True,
        padding=False,
    )

    # Label masking: ignore pad tokens in loss calculation
    cleaned_labels = []
    for label_seq in labels["input_ids"]:
        cleaned = [
            token_id if token_id != tokenizer.pad_token_id else -100
            for token_id in label_seq
        ]
        cleaned_labels.append(cleaned)

    model_inputs["labels"] = cleaned_labels
    return model_inputs


def build_compute_metrics_fn(tokenizer):
    """Evaluation metrics for SQL generation (Exact Match and BLEU)."""
    bleu = evaluate.load("bleu")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Normalize negative padding IDs for decoding
        preds = np.where((preds < 0) | (preds == -100), tokenizer.pad_token_id, preds)
        labels = np.where((labels < 0) | (labels == -100), tokenizer.pad_token_id, labels)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        exact_matches = [
            int(p.strip().lower() == l.strip().lower())
            for p, l in zip(decoded_preds, decoded_labels)
        ]
        exact_match_score = float(np.mean(exact_matches)) if exact_matches else 0.0

        try:
            bleu_res = bleu.compute(
                predictions=[p.strip() for p in decoded_preds],
                references=[[l.strip()] for l in decoded_labels],
            )
            bleu_score = float(bleu_res.get("bleu", 0.0))
        except Exception:
            bleu_score = 0.0

        return {
            "exact_match": exact_match_score,
            "bleu": bleu_score,
        }

    return compute_metrics


def train(
    train_file: Path,
    model_name: str = "google/flan-t5-base",
    output_dir: Path = Path("training/artifacts/flan_t5_sql_lora"),
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    learning_rate: float = 3e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_rslora: bool = True,
    use_dora: bool = False,
    eval_ratio: float = 0.05,
    max_source_len: int = 512,
    max_target_len: int = 256,
    seed: int = 42,
) -> None:
    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_file}")

    logger.info("Loading dataset from %s", train_file)
    dataset = load_dataset("json", data_files=str(train_file), split="train")
    dataset = dataset.shuffle(seed=seed)
    split_dataset = dataset.train_test_split(test_size=eval_ratio, seed=seed)

    logger.info("Initializing tokenizer and base model (%s)", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Technique: Target all linear attention and feedforward layers
    target_modules = ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
    # Check layer names present in the model
    available_modules = {name.split(".")[-1] for name, _ in base_model.named_modules()}
    valid_targets = [m for m in target_modules if m in available_modules]
    if not valid_targets:
        valid_targets = ["q", "v"]  # Fallback

    logger.info("Applying LoRA config (r=%d, alpha=%d, rsLoRA=%s, DoRA=%s, targets=%s)",
                lora_r, lora_alpha, use_rslora, use_dora, valid_targets)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=valid_targets,
        use_rslora=use_rslora,
        use_dora=use_dora,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    logger.info("Tokenizing train and eval splits...")
    tokenized_train = split_dataset["train"].map(
        lambda b: preprocess_batch(b, tokenizer, max_source_len, max_target_len),
        batched=True,
        remove_columns=split_dataset["train"].column_names,
    )
    tokenized_eval = split_dataset["test"].map(
        lambda b: preprocess_batch(b, tokenizer, max_source_len, max_target_len),
        batched=True,
        remove_columns=split_dataset["test"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Hardware detection: fp16/bf16
    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16

    logger.info("Training device: %s (fp16=%s, bf16=%s)", "CUDA" if has_cuda else "CPU", use_fp16, use_bf16)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=25,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(tokenizer),
    )

    logger.info("Starting training run...")
    trainer.train()

    # Save final LoRA adapter
    final_output = output_dir / "final"
    final_output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    logger.info("Training complete! Saved fine-tuned adapter to: %s", final_output)


def main():
    parser = argparse.ArgumentParser(description="Advanced Fine-Tuning for FLAN-T5 Text-to-SQL")
    parser.add_argument("--train-file", type=Path, default=Path("training/data/mixture_train.jsonl"))
    parser.add_argument("--model", type=str, default="google/flan-t5-base")
    parser.add_argument("--output-dir", type=Path, default=Path("training/artifacts/flan_t5_sql_lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--use-dora", action="store_true", help="Enable Weight-Decomposed LoRA")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--max-source-len", type=int, default=512)
    parser.add_argument("--max-target-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        train_file=args.train_file,
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_rslora=True,
        use_dora=args.use_dora,
        eval_ratio=args.eval_ratio,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
