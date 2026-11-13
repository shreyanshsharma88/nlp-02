"""Model loading (requirement 3) and fine-tuning (requirement 4). One function trains either model with the same config."""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass

import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

from ner.evaluate import make_trainer_compute_metrics
from ner.tokenize import tokenize_dataset

MODELS = ["bert-base-cased", "bert-base-uncased"]


@dataclass
class TrainConfig:
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    seed: int = 42
    logging_steps: int = 50


def device_name(use_cpu: bool = False) -> str:
    if use_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


def build_model(model_name: str, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    return tokenizer, model


def run_name(model_name: str, seed: int) -> str:
    return f"{model_name}_s{seed}"


def train_model(model_name: str, ds, id2label, label2id, out_root: str, logs_dir: str,
                cfg: TrainConfig = TrainConfig(), use_cpu: bool = False, force: bool = False,
                run_label: str | None = None):
    """Fine-tune `model_name`; return summary dict. Resumable: skips if the best model + log already exist.
    `run_label` names the run directory (defaults to the model name), e.g. for the case-augmentation extra."""
    name = run_name(run_label or model_name, cfg.seed)
    best_dir = os.path.join(out_root, name, "best")
    log_path = os.path.join(logs_dir, f"{name}.json")
    if not force and os.path.exists(os.path.join(best_dir, "config.json")) and os.path.exists(log_path):
        print(f"[skip] {name} already trained")
        with open(log_path) as f:
            return json.load(f)

    set_seed(cfg.seed)
    tokenizer, model = build_model(model_name, id2label, label2id)
    tok_ds = tokenize_dataset(ds, tokenizer, cfg.max_length)
    collator = DataCollatorForTokenClassification(tokenizer)
    use_cuda = torch.cuda.is_available() and not use_cpu
    args = TrainingArguments(
        output_dir=os.path.join(out_root, name),
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_ratio,  # transformers>=5: float in [0,1) = warmup ratio
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        # Best checkpoint is chosen by validation entity F1. We do NOT use load_best_model_at_end: in
        # transformers 5.x the Trainer reloads checkpoints without the gamma/beta -> weight/bias key
        # conversion, silently leaving LayerNorm params from the last epoch. We reload with from_pretrained.
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=None,
        logging_steps=cfg.logging_steps,
        seed=cfg.seed,
        data_seed=cfg.seed,
        fp16=use_cuda,
        use_cpu=use_cpu,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tok_ds["train"], eval_dataset=tok_ds["validation"],
                      processing_class=tokenizer, data_collator=collator,
                      compute_metrics=make_trainer_compute_metrics(id2label))
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    best_ckpt = trainer.state.best_model_checkpoint
    assert best_ckpt, "no best checkpoint recorded"
    best_model = AutoModelForTokenClassification.from_pretrained(best_ckpt)
    best_model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    eval_trainer = Trainer(model=best_model, args=args, eval_dataset=tok_ds["validation"], processing_class=tokenizer,
                           data_collator=collator, compute_metrics=make_trainer_compute_metrics(id2label))
    val = eval_trainer.evaluate()
    assert abs(val["eval_f1"] - trainer.state.best_metric) < 1e-6, (val["eval_f1"], trainer.state.best_metric)
    for d in os.listdir(os.path.join(out_root, name)):  # drop intermediate checkpoints (~1.2 GB each)
        if d.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(out_root, name, d), ignore_errors=True)
    summary = {
        "model": model_name, "run": name, "seed": cfg.seed, "config": asdict(cfg), "device": device_name(use_cpu),
        "fp16": use_cuda, "train_time_sec": train_time, "train_sentences": len(ds["train"]),
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_metric_val_f1": trainer.state.best_metric,
        "val_metrics_best": val, "log_history": trainer.state.log_history,
        "torch": torch.__version__,
    }
    os.makedirs(logs_dir, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    del trainer, model, best_model, eval_trainer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return summary
