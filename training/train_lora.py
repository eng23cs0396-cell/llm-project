from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def build_prompt(instruction: str, model_input: str) -> str:
    return f"Instruction: {instruction}\n{model_input}\nSQL:"


def tokenize_function(batch, tokenizer, max_source_len: int, max_target_len: int):
    prompts = [build_prompt(i, x) for i, x in zip(batch["instruction"], batch["input"])]
    targets = batch["output"]

    model_inputs = tokenizer(
        prompts,
        max_length=max_source_len,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_len,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics_fn(tokenizer):
    bleu = evaluate.load("bleu")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Some trainer outputs may include negative padding ids; normalize before decode.
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        exact = np.mean(
            [
                int(pred.strip().lower() == label.strip().lower())
                for pred, label in zip(decoded_preds, decoded_labels)
            ]
        )
        bleu_result = bleu.compute(
            predictions=[p.strip() for p in decoded_preds],
            references=[[l.strip()] for l in decoded_labels],
        )
        return {
            "exact_match": float(exact),
            "bleu": float(bleu_result["bleu"]),
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for NL to SQL")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--model", default="google/flan-t5-base")
    parser.add_argument("--output-dir", type=Path, default=Path("training/artifacts/flan_t5_sql_lora"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-source-len", type=int, default=512)
    parser.add_argument("--max-target-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.train_file.exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")

    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    dataset = dataset.shuffle(seed=args.seed)
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v"],
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

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_steps=300,
        eval_strategy="steps",
        eval_steps=300,
        save_total_limit=2,
        predict_with_generate=True,
        report_to="none",
        fp16=False,
        bf16=False,
        gradient_accumulation_steps=1,
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(tokenizer),
    )

    trainer.train()

    # Always persist a final checkpoint even if metric computation fails.
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    try:
        metrics = trainer.evaluate(max_length=args.max_target_len)
        print("Final eval metrics:", metrics)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Evaluation metrics failed after training: {exc}")

    print(f"Saved adapter/model to {final_dir}")


if __name__ == "__main__":
    main()
