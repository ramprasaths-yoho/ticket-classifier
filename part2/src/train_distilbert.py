"""Fine-tune DistilBERT with the Trainer API for ONE target (category or priority).

Run from the project root:
    python src/train_distilbert.py --target category
    python src/train_distilbert.py --target priority
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "distilbert-base-uncased"


class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["category", "priority"], required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)

    data = Path(args.data_dir)
    train = pd.read_csv(data / "train.csv")
    val = pd.read_csv(data / "val.csv")
    test = pd.read_csv(data / "test.csv")

    labels = sorted(train[args.target].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def make(df):
        return TicketDataset(df["text"], df[args.target].map(label2id), tok, args.max_length)

    train_ds, val_ds, test_ds = make(train), make(val), make(test)

    out = Path(args.out_dir) / f"distilbert_{args.target}"
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - t0

    val_m = trainer.evaluate(val_ds, metric_key_prefix="val")
    test_m = trainer.evaluate(test_ds, metric_key_prefix="test")

    trainer.save_model(str(out))  # model + config (with id2label)
    tok.save_pretrained(str(out))
    (out / "metrics.json").write_text(
        json.dumps({"train_seconds": train_seconds, "val": val_m, "test": test_m}, indent=2)
    )
    print(f"\nSaved to {out}\ntrain time: {train_seconds:.1f}s")
    print(f"val  acc={val_m['val_accuracy']:.3f} macroF1={val_m['val_macro_f1']:.3f}")
    print(f"test acc={test_m['test_accuracy']:.3f} macroF1={test_m['test_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
