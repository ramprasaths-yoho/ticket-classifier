# Comparison - category

## Test set

| Model | Accuracy (test) | Macro F1 (test) | Training time | Inference / ticket |
|---|---|---|---|---|
| TF-IDF + LogReg | 1.000 | 1.000 | 0.1 s | 2.25 ms |
| DistilBERT | 0.970 | 0.968 | 74.2 s | 41.62 ms |

## Validation set

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + LogReg | 1.000 | 1.000 |
| DistilBERT | 1.000 | 1.000 |

## Per-class F1 (test)

| Class | Baseline | DistilBERT | Diff |
|---|---|---|---|
| Account | 1.000 | 1.000 | +0.000 |
| Billing | 1.000 | 1.000 | +0.000 |
| Cancellation | 1.000 | 1.000 | +0.000 |
| Shipping | 1.000 | 0.933 | -0.067 |
| Technical | 1.000 | 0.909 | -0.091 |
