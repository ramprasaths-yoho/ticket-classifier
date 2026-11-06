# Customer Support Ticket Classifier - Part 2

Fine-tuned DistilBERT vs TF-IDF baseline, served with FastAPI.

## Project layout
```
data/            train.csv, val.csv, test.csv   (columns: text, category, priority)
models/          baseline_category.joblib, baseline_priority.joblib  (from Part 1)
                 distilbert_category/, distilbert_priority/          (created by training)
src/             train_distilbert.py, compare_models.py
app/             main.py (FastAPI), model.py (loading + prediction)
tests/           test_api.py
results/         comparison_<target>.md
```

## Setup (VS Code terminal)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train DistilBERT
```bash
python src/train_distilbert.py --target category
python src/train_distilbert.py --target priority     # if time allows
```

## 2. Compare with the baseline
```bash
python src/compare_models.py --target category
```
Results are written to `results/comparison_category.md`.

| Model | Accuracy | Macro F1 | Training time | Inference / ticket |
|---|---|---|---|---|
| TF-IDF + LogReg | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| DistilBERT | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

### Deployment choice (5 lines)
_Write from your own numbers: accuracy/F1 gap, per-class wins, training + inference cost, hardware (CPU/GPU), and final pick._

## 3. Run the API
```bash
uvicorn app.main:app --reload
# use the baseline instead:  MODEL_BACKEND=baseline uvicorn app.main:app
```
Docs: http://127.0.0.1:8000/docs

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice"}'
# {"category":"Billing","priority":"High","confidence":0.93}

# empty text -> 422
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text": ""}'

# bonus: batch
curl -X POST http://127.0.0.1:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '[{"text": "I was charged twice"}, {"text": "App crashes on login"}]'
```
(PowerShell: use `curl.exe` and escape quotes, or use the `/docs` page.)

## 4. Tests
```bash
pytest -v
```
Tests use a fake predictor, so they run without trained models.

## Git / PR
```bash
git checkout -b feature/distilbert-fastapi
git add . && git commit -m "Part 2: DistilBERT fine-tuning, comparison, FastAPI endpoint"
git push -u origin feature/distilbert-fastapi
# open a Pull Request on GitHub, get review, merge
```
Add `models/` and `.venv/` to `.gitignore` (DistilBERT weights are ~250 MB).
