"""Compare the TF-IDF baseline (Part 1) with fine-tuned DistilBERT on the SAME val/test sets.

Writes results/comparison.md (table + per-class F1 analysis).
    python src/compare_models.py --target category
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def per_ticket_ms(predict_one, texts, n=100):
    texts = list(texts)[:n]
    predict_one(texts[0])  # warm-up
    t0 = time.perf_counter()
    for t in texts:
        predict_one(t)
    return (time.perf_counter() - t0) / len(texts) * 1000


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["category", "priority"], default="category")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--baseline", default=None, help="default: models/baseline_<target>.joblib")
    args = p.parse_args()

    d = Path(args.data_dir)
    train, val, test = (pd.read_csv(d / f"{s}.csv") for s in ("train", "val", "test"))
    y_val, y_test = val[args.target].tolist(), test[args.target].tolist()

    # ---- Baseline (TF-IDF + LogReg) ----
    base_path = args.baseline or f"{args.models_dir}/baseline_{args.target}.joblib"
    base = joblib.load(base_path)
    t0 = time.perf_counter()
    clone(base).fit(train["text"], train[args.target])  # re-fit only to time training
    base_train = time.perf_counter() - t0
    b_val = base.predict(val["text"])
    b_test = base.predict(test["text"])
    b_ms = per_ticket_ms(lambda t: base.predict([t]), test["text"])

    # ---- DistilBERT ----
    mdir = Path(args.models_dir) / f"distilbert_{args.target}"
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).eval()
    id2label = model.config.id2label

    @torch.no_grad()
    def bert_predict(texts):
        enc = tok(list(texts), truncation=True, max_length=128, padding=True, return_tensors="pt")
        return [id2label[i] for i in model(**enc).logits.argmax(-1).tolist()]

    def bert_batches(texts, bs=32):
        texts = list(texts)
        return sum((bert_predict(texts[i : i + bs]) for i in range(0, len(texts), bs)), [])

    d_val, d_test = bert_batches(val["text"]), bert_batches(test["text"])
    d_ms = per_ticket_ms(lambda t: bert_predict([t]), test["text"])
    import json
    d_train = json.loads((mdir / "metrics.json").read_text())["train_seconds"]

    def row(name, yv, pv, yt, pt, train_s, ms):
        return [
            name,
            f"{accuracy_score(yt, pt):.3f}",
            f"{f1_score(yt, pt, average='macro'):.3f}",
            f"{train_s:.1f} s",
            f"{ms:.2f} ms",
        ]

    headers = ["Model", "Accuracy (test)", "Macro F1 (test)", "Training time", "Inference / ticket"]
    rows = [
        row("TF-IDF + LogReg", y_val, b_val, y_test, b_test, base_train, b_ms),
        row("DistilBERT", y_val, d_val, y_test, d_test, d_train, d_ms),
    ]
    val_rows = [
        ["TF-IDF + LogReg", f"{accuracy_score(y_val, b_val):.3f}", f"{f1_score(y_val, b_val, average='macro'):.3f}"],
        ["DistilBERT", f"{accuracy_score(y_val, d_val):.3f}", f"{f1_score(y_val, d_val, average='macro'):.3f}"],
    ]

    # ---- Where does each win? per-class F1 on test ----
    rb = classification_report(y_test, b_test, output_dict=True, zero_division=0)
    rd = classification_report(y_test, d_test, output_dict=True, zero_division=0)
    classes = sorted(set(y_test))
    cls_rows = [
        [c, f"{rb[c]['f1-score']:.3f}", f"{rd[c]['f1-score']:.3f}", f"{rd[c]['f1-score'] - rb[c]['f1-score']:+.3f}"]
        for c in classes
    ]

    out = Path("results")
    out.mkdir(exist_ok=True)
    text = (
        f"# Comparison - {args.target}\n\n## Test set\n\n{md_table(headers, rows)}\n\n"
        f"## Validation set\n\n{md_table(['Model', 'Accuracy', 'Macro F1'], val_rows)}\n\n"
        f"## Per-class F1 (test)\n\n{md_table(['Class', 'Baseline', 'DistilBERT', 'Diff'], cls_rows)}\n"
    )
    (out / f"comparison_{args.target}.md").write_text(text)
    print(text)

    # Misclassifications DistilBERT fixes (useful for the analysis write-up)
    wrong_b = np.array(b_test) != np.array(y_test)
    right_d = np.array(d_test) == np.array(y_test)
    fixed = test[wrong_b & right_d]
    print(f"\nTickets baseline got wrong but DistilBERT got right: {len(fixed)}")
    print(fixed["text"].head(8).to_string())


if __name__ == "__main__":
    main()
