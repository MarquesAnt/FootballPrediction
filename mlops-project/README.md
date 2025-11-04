# MLOps Project Template

Pipeline (DVC) → Model training → API (FastAPI) → Docker → CI/CD (GitHub Actions) → Monitoring (Prometheus/Evidently).

## Quickstart

```bash
# 1) Python env
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 2) Install deps
pip install -r requirements.txt
pre-commit install

# 3) Init DVC
dvc init
# (Optional) Configure remote, e.g. S3:
# dvc remote add -d s3 s3://<bucket>/<path>
# dvc remote modify s3 access_key_id <ID>
# dvc remote modify s3 secret_access_key <SECRET>

# 4) Reproduce pipeline (dummy for now)
dvc repro

# 5) Run API
uvicorn src.api.app:app --reload --port 8000
```

## Project layout
```
mlops-project/
├─ data/                      # DVC-tracked (raw/, processed/) -> .gitignored
├─ dvc.yaml                   # pipeline DVC
├─ params.yaml                # hyperparams and feature config
├─ src/
│  ├─ data/ingest.py          # download / make demo data
│  ├─ data/make_features.py   # feature engineering
│  ├─ models/train.py         # training + metrics
│  ├─ models/predict.py       # batch/online inference
│  ├─ utils/                  # helpers
│  └─ api/app.py              # FastAPI app
├─ models/                    # artifacts (DVC)
├─ reports/                   # metrics, plots (DVC)
├─ tests/                     # pytest (unit + e2e)
├─ .github/workflows/ci.yml   # GitHub Actions
├─ Dockerfile                 # container
└─ requirements.txt
```

## Endpoints
- `GET /health` — service health
- `GET /metrics` — Prometheus metrics
- `POST /predict` — simple demo model (mean baseline)

You can now plug your real use case and data.
