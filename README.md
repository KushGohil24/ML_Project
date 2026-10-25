# Fake Job Posting Detection System

> **College ML Project** · MERN Stack + Python ML Pipeline

A complete end-to-end system that classifies job postings as **real** or **fraudulent** using 9 ML models, TF-IDF text features, and a MERN web application.

---

## 📁 Project Structure

```
fake-job-detector/
├── ml/
│   ├── data/            # fake_job_postings.csv (EMSCAD dataset)
│   ├── src/             # Python ML modules
│   │   ├── preprocess.py   # Data cleaning + feature engineering
│   │   ├── features.py     # TF-IDF + one-hot feature matrix
│   │   ├── eda.py          # EDA charts + stats
│   │   ├── models.py       # All 9 ML model training
│   │   ├── unsupervised.py # SVD, K-Means, SOM
│   │   ├── interpret.py    # Feature importance + SHAP
│   │   └── train.py        # Main pipeline orchestrator
│   ├── models/          # Saved models + metrics.json
│   ├── exports/         # Generated charts (PNG) + curve JSON
│   ├── service.py       # FastAPI inference service
│   ├── requirements.txt
│   └── venv/            # Python virtual environment
├── server/              # Express backend
│   ├── models/          # Mongoose schemas
│   ├── routes/          # API route handlers
│   ├── server.js
│   ├── seed.js
│   └── package.json
├── client/              # React + Vite frontend
│   └── src/
│       ├── pages/       # 7 pages
│       ├── api/         # Axios API layer
│       └── index.css    # Design system
└── README.md
```

---

## 🚀 Run Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB running locally (or Atlas URI)

---

### Step 1 — ML Training Pipeline

```bash
cd fake-job-detector/ml
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Go back to fake-job-detector/ to run the pipeline
cd ..
python ml/src/train.py
```

**Expected output:**
- `ml/models/best_model.joblib` — best ML model
- `ml/models/tfidf.joblib` + `ml/models/ohe.joblib` — fitted vectorizers
- `ml/models/metrics.json` — all model metrics
- `ml/exports/*.png` — 15+ charts
- `ml/exports/eda_stats.json` — EDA stats

Training takes **~10-30 minutes** depending on your hardware.

---

### Step 2 — Python ML Service (FastAPI)

```bash
# From fake-job-detector/
source ml/venv/bin/activate
uvicorn ml.service:app --port 8000 --reload
```

Test it: `curl http://localhost:8000/health`

---

### Step 3 — Express Backend

```bash
cd server
cp .env.example .env   # edit MONGO_URI if needed
npm install
node seed.js           # import metrics into MongoDB
npm start              # or: npm run dev (with nodemon)
```

Test it: `curl http://localhost:5000/api/health`

---

### Step 4 — React Frontend

```bash
cd client
npm install
npm run dev
```

Open: **http://localhost:5173**

---

### 🏃 Start Everything At Once (Single Command)

You can launch all three services simultaneously with our unified runner:

```bash
# Option A: Bash script
./start.sh

# Option B: npm script
npm start
```

This starts:
- 🐍 **Python ML Service** on `http://localhost:8000`
- 🟢 **Express API Backend** on `http://localhost:5000`
- ⚛️ **React Frontend** on `http://localhost:5173`

Or run them individually in separate terminals:
```bash
# Terminal 1 — ML Service
source ml/venv/bin/activate && uvicorn ml.service:app --port 8000

# Terminal 2 — Express Server
cd server && npm start

# Terminal 3 — React Dev Server
cd client && npm run dev
```

---

## 📓 Jupyter Notebook

A complete, beginner-friendly pipeline walkthrough notebook is provided:
- **`ml/notebooks/fake_job_detection_pipeline.ipynb`**
Includes step-by-step code, Markdown explanations, mathematical concepts, visualizations, and college viva questions.

---

## 📊 Model Results

| Model | Precision | Recall | F1 | PR-AUC |
|-------|-----------|--------|----|--------|
| SVM | ~0.99 | ~0.75 | ~0.86 | ~0.94 |
| Random Forest | ~0.99 | ~0.67 | ~0.80 | ~0.93 |
| XGBoost | ~0.97 | ~0.74 | ~0.84 | ~0.94 |
| Logistic Regression | ~0.42 | ~0.87 | ~0.56 | ~0.78 |
| Decision Tree (tuned) | ~0.50 | ~0.79 | ~0.62 | ~0.57 |

*Exact numbers from your training run are in `ml/models/metrics.json`.*

---

## 🔑 Environment Variables

**`server/.env`**
```
MONGO_URI=mongodb://localhost:27017/fake_job_detector
ML_SERVICE_URL=http://localhost:8000
PORT=5000
```

**`client/.env`**
```
VITE_API_URL=http://localhost:5000/api
```

---

## 📚 Key Concepts

| Concept | What it does in this project |
|---------|------------------------------|
| TF-IDF | Converts job text → 10k numeric features |
| SMOTE | Synthesizes fraud examples to fix 5% imbalance |
| PR-AUC | Main metric (handles class imbalance) |
| SHAP | Explains WHY a posting was flagged |
| SVD/PCA | Reduces features → 2D for visualization |
| K-Means | Clusters job postings unsupervised |
| SOM | Neural network topology map of postings |

---

## 🚀 Future Scope

1. **BERT Embeddings** — context-aware text representations
2. **Reinforcement Learning** — agent asks clarifying questions
3. **Job Portal Integration** — browser extension for LinkedIn/Indeed
4. **Graph Neural Networks** — detect fraud rings

---

*Dataset: EMSCAD Fake Job Postings (Kaggle) | 17,880 rows | 4.8% fraud*
