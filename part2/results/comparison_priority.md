# Comparison - priority

## Test set

| Model | Accuracy (test) | Macro F1 (test) | Training time | Inference / ticket |
|---|---|---|---|---|
| TF-IDF + LogReg | 1.000 | 1.000 | 0.1 s | 0.77 ms |
| DistilBERT | 1.000 | 1.000 | 67.8 s | 38.28 ms |

## Validation set

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + LogReg | 0.939 | 0.947 |
| DistilBERT | 0.970 | 0.973 |

## Per-class F1 (test)

| Class | Baseline | DistilBERT | Diff |
|---|---|---|---|
| High | 1.000 | 1.000 | +0.000 |
| Low | 1.000 | 1.000 | +0.000 |
| Medium | 1.000 | 1.000 | +0.000 |
