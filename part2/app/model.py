"""Model loading + prediction. Instantiated ONCE at app startup (see app/main.py)."""
import os
from pathlib import Path

import joblib

TARGETS = ("category", "priority")
MAX_LENGTH = 128


class BasePredictor:
    def _head(self, target, texts):
        """Return [(label, confidence), ...] for one target."""
        raise NotImplementedError

    def predict(self, texts):
        cats = self._head("category", texts)
        pris = self._head("priority", texts)
        return [
            {"category": c, "priority": p, "confidence": round(conf, 4)}
            for (c, conf), (p, _) in zip(cats, pris)
        ]


class DistilBertPredictor(BasePredictor):
    def __init__(self, models_dir):
        # imported lazily so tests / the baseline backend don't need torch
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.heads = {}
        for t in TARGETS:
            path = Path(models_dir) / f"distilbert_{t}"
            if path.exists():
                tok = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path).eval()
                self.heads[t] = (tok, model)
        if "category" not in self.heads:
            raise FileNotFoundError(f"No fine-tuned category model in {models_dir}/distilbert_category")

    def _head(self, target, texts):
        if target not in self.heads:
            return [(None, 0.0)] * len(texts)
        tok, model = self.heads[target]
        enc = tok(list(texts), truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        with self._torch.no_grad():
            probs = self._torch.softmax(model(**enc).logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return [(model.config.id2label[i], float(c)) for i, c in zip(idx.tolist(), conf)]


class BaselinePredictor(BasePredictor):
    def __init__(self, models_dir):
        self.pipes = {}
        for t in TARGETS:
            path = Path(models_dir) / f"baseline_{t}.joblib"
            if path.exists():
                self.pipes[t] = joblib.load(path)
        if "category" not in self.pipes:
            raise FileNotFoundError(f"No baseline category model in {models_dir}/baseline_category.joblib")

    def _head(self, target, texts):
        pipe = self.pipes.get(target)
        if pipe is None:
            return [(None, 0.0)] * len(texts)
        probs = pipe.predict_proba(list(texts))
        idx = probs.argmax(axis=1)
        return [(str(pipe.classes_[i]), float(probs[r, i])) for r, i in enumerate(idx)]


def load_predictor():
    """MODEL_BACKEND=distilbert (default) or baseline; MODELS_DIR=models (default)."""
    backend = os.getenv("MODEL_BACKEND", "distilbert").lower()
    models_dir = os.getenv("MODELS_DIR", "models")
    return BaselinePredictor(models_dir) if backend == "baseline" else DistilBertPredictor(models_dir)
