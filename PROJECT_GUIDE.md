# 📘 Complete College Project Master Guide: Fake Job Posting Detection System

> **A Comprehensive, Plain-English Guide for Students & Viva Preparation**  
> *Course: Semester 7 Machine Learning Project*  
> *Stack: Python (scikit-learn, XGBoost, TF-IDF, SMOTE, MiniSom) + MERN Stack (MongoDB, Express, React 19, Node.js)*

---

## 📑 Table of Contents
1. [Project Overview in Plain English](#1-project-overview-in-plain-english)
2. [The Core Problem & The "Accuracy Trap"](#2-the-core-problem--the-accuracy-trap)
3. [The Dataset (EMSCAD) Breakdown](#3-the-dataset-emscad-breakdown)
4. [Step-by-Step Data Preprocessing & Feature Engineering](#4-step-by-step-data-preprocessing--feature-engineering)
5. [Preventing Data Leakage & SMOTE Explained](#5-preventing-data-leakage--smote-explained)
6. [The 9 Machine Learning Models Explained Simply](#6-the-9-machine-learning-models-explained-simply)
7. [Benchmark Results & Why Linear SVM Won](#7-benchmark-results--why-linear-svm-won)
8. [Unsupervised Learning (SVD, K-Means, SOM)](#8-unsupervised-learning-svd-k-means-som)
9. [Model Explainability & Structural Red Flags](#9-model-explainability--structural-red-flags)
10. [Full-Stack Architecture & Code Flow](#10-full-stack-architecture--code-flow)
11. [How to Run and Demo the Project](#11-how-to-run-and-demo-the-project)
12. [College Viva Voce Top 12 Q&A Cheat Sheet](#12-college-viva-voce-top-12-qa-cheat-sheet)

---

## 1. Project Overview in Plain English

### What does this project do?
This system takes an online job posting (its title, description, company profile, salary, requirements, and metadata) and **automatically classifies whether it is legitimate (real) or a scam (fraudulent)**.

### Why is this needed?
Scammers post fake jobs on portals like LinkedIn, Indeed, and Telegram to steal personal identification details (SSN, passport, bank accounts) or demand advance fees ("pay $50 for a background check / starter kit"). Manual review of thousands of listings daily is impossible for job portals, so an automated, explainable ML classifier is essential.

### How is it built?
It is built as two collaborating layers:
1. **Machine Learning Pipeline (Python)**:
   - Trained offline on 17,880 historical job postings.
   - Cleans raw text, extracts 10,810 TF-IDF and domain features, balances the dataset using SMOTE, and evaluates 9 models.
   - Saves the champion model (`SVM`) and serves live predictions via a lightweight **FastAPI** service on port `8000`.
2. **Web Application (MERN Stack)**:
   - **React 19 Frontend (Vite)** on port `5173`: A modern, clean, light-themed analytics dashboard where users can test job postings, view model benchmarks, explore dataset statistics, and review past audit logs.
   - **Express + Node.js Backend** on port `5001`: Validates input, relays requests to the Python ML microservice, and stores predictions in **MongoDB**.

---

## 2. The Core Problem & The "Accuracy Trap"

### What is Class Imbalance?
In our dataset of 17,880 jobs:
- **17,014 are Legitimate (95.16%)**
- **866 are Fraudulent (4.84%)**

There is roughly **1 fake job for every 20 real jobs**.

### Why is Accuracy a completely misleading metric here?
Imagine a lazy model that **always guesses "Real"** for every single posting without even reading the text:
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{17,014}{17,880} = 95.16\%$$
The model has **95.2% accuracy**, but it caught **0 fake jobs**! It is 100% useless in the real world.

### What metrics do we use instead?
- **Precision**: When the model flags a job as *Fraudulent*, how often is it actually right? (High precision means genuine recruiters don't get falsely accused).
- **Recall**: Out of all actual fake jobs out there, what percentage did the model catch? (High recall means scammers don't slip through).
- **F1-Score**: The harmonic mean of Precision and Recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **PR-AUC (Precision-Recall Area Under Curve)**: **Our primary gold standard.** It evaluates Precision and Recall across all decision thresholds specifically focusing on the rare minority class (fraud).

---

## 3. The Dataset (EMSCAD) Breakdown

We use the **Employment Scam Aegean Dataset (EMSCAD)** created by the University of the Aegean, containing 17,880 annotated job listings with 18 fields:

| Field Group | Column Names | What it Represents |
|---|---|---|
| **Text Fields** | `title`, `company_profile`, `description`, `requirements`, `benefits` | Raw unstructured text of the job description. |
| **Categorical Metadata** | `employment_type`, `required_experience`, `required_education`, `salary_range` | Structured dropdown selections (e.g. Full-time, Bachelor's degree). |
| **Binary Flags** | `has_company_logo`, `has_questions`, `telecommuting` | 1 if present, 0 if absent. |
| **Target Label** | `fraudulent` | `0` = Legitimate, `1` = Fraudulent. |

---

## 4. Step-by-Step Data Preprocessing & Feature Engineering

Computers cannot process raw English sentences directly; text must be mathematically transformed into numerical vectors.

```
[Raw Job Posting]
       │
       ▼
1. HTML Stripping & Regex Cleaning (removes <p>, URLs, pure numbers)
       │
       ▼
2. Stopword Removal & Lemmatization ("running" -> "run", removes "the", "and")
       │
       ▼
3. TF-IDF Vectorization (10,000 unigrams & bigrams like "wire transfer", "starter kit")
       │
       ▼
4. Domain Feature Extraction (missing logo, missing salary, scam trigger words)
       │
       ▼
5. Feature Stacking (10,000 text features + 800 metadata categories + 10 numeric flags)
       │
       ▼
[Final Matrix: 10,810 Numeric Columns per Job]
```

### 1. Text Cleaning (`ml/src/preprocess.py`)
- Removes HTML tags (`<br>`, `<b>`).
- Removes URLs (`https://...`) and emails.
- Converts everything to lowercase.
- Lemmatizes words (converts "hiring", "hires", "hired" to root form "hire").

### 2. TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF calculates how unique and important a word or phrase is:
- Common words across all jobs like "team", "work", "responsibilities" get a **low** score.
- Words distinctive to scams like "wire transfer", "western union", "starter kit", "processing fee" get a **high** score.
- We use **1-2 n-grams** (single words and two-word pairs) up to the top **10,000 features**.

### 3. Domain Feature Engineering
Scammers share distinct behavior patterns outside of vocabulary:
- `missing_logo`: Scammers rarely upload authentic corporate logos (80%+ of scam jobs have no logo).
- `missing_salary`: Undisclosed compensation.
- `has_questions`: Legitimate companies almost always ask screening questions; scammers skip them to lure victims quickly.
- `scam_keyword_count`: Frequency of high-risk terms ("wire transfer", "money gram", "earn $5000", "be your own boss").

---

## 5. Preventing Data Leakage & SMOTE Explained

### What is Data Leakage?
Data leakage occurs when information from the test dataset leaks into the training pipeline before the model is tested, giving artificially high, fake test results.

### How we prevented Data Leakage:
1. **Split First**: We perform an 80/20 Stratified Split (`train_test_split`) before any vectorization or scaling.
2. **Fit on Train Only**: The TF-IDF vectorizer and categorical encoders are `.fit()` on the training set only, and merely `.transform()` the test set.
3. **SMOTE on Train Only**: Synthetic oversampling was strictly applied to the training set. The test set was **never touched** so it mirrors realistic real-world 95:5 test conditions.

### What is SMOTE (Synthetic Minority Over-sampling Technique)?
Instead of just duplicating existing fake jobs (which causes overfitting), SMOTE finds a fake job in feature space, looks at its $k$ nearest fake job neighbors, and mathematically interpolates brand new synthetic points along the connecting line.
- **Before SMOTE**: 13,611 real jobs vs 693 fake jobs (in training split).
- **After SMOTE**: 13,611 real jobs vs 13,611 synthetic fake jobs (balanced 50:50).

---

## 6. The 9 Machine Learning Models Explained Simply

We implemented 3 distinct tiers of machine learning models to see how performance scales:

### Tier 1: Baseline Classifiers
1. **Logistic Regression**: A linear model using a sigmoid function to output a probability between 0 and 1. Highly interpretable baseline.
2. **K-Nearest Neighbors (KNN)**: Classifies a job based on the majority vote of its 5 closest jobs in feature space. Performs poorly in 10,000 dimensions due to the "curse of dimensionality".
3. **Decision Tree (Overfitting Demo)**: An unconstrained tree grown until every training point is pure. Shows a 100% training F1 score but drops to 71% on test data (proves overfitting).
4. **Decision Tree (Tuned)**: Constrained with `max_depth=10` and `min_samples_leaf=5` to prevent memorizing noise.
5. **Support Vector Machine (LinearSVC + Platt Calibration)**: Finds the maximum-margin hyperplane that separates real from fake jobs with strong L2 regularization.

### Tier 2: Ensemble Models
6. **Random Forest**: Builds 150 independent decision trees on random data subsets and feature subsets, averaging their votes (bagging).
7. **Gradient Boosting**: Builds trees sequentially, where each new tree specifically tries to correct the residual errors made by the previous trees.
8. **XGBoost (Extreme Gradient Boosting)**: Highly optimized, regularized gradient boosting algorithm using second-order gradients and column subsampling.

### Tier 3: Deep Neural Networks
9. **Keras Feed-Forward MLP & 1D-CNN**: Multi-layer perceptron with Dense layers (256 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 1), ReLU activation, Dropout (0.4), and Early Stopping.

---

## 7. Benchmark Results & Why Linear SVM Won

Evaluated strictly on the held-out test set (3,576 postings):

| Model | Precision | Recall | F1-Score | ROC-AUC | **PR-AUC (Main Metric)** | Verdict |
|---|---|---|---|---|---|---|
| **Support Vector Machine (LinearSVC)** | **0.9924** | **0.7514** | **0.8553** | **0.9909** | **0.9361** | 🏆 **Champion Model** |
| **Random Forest** | 0.9912 | 0.6647 | 0.7960 | 0.9918 | 0.9341 | Strong ensemble |
| **XGBoost** | 0.9154 | 0.6879 | 0.7852 | 0.9859 | 0.8872 | High performance |
| **Gradient Boosting** | 0.7744 | 0.7514 | 0.7621 | 0.9822 | 0.8369 | Balanced precision/recall |
| **Decision Tree (Tuned)** | 0.5037 | 0.7919 | 0.6157 | 0.9134 | 0.5694 | Regularized tree |
| **Logistic Regression** | 0.4160 | 0.8728 | 0.5634 | 0.9708 | 0.7788 | High recall, low precision |
| **Decision Tree (Overfit)** | 0.6522 | 0.7803 | 0.7105 | 0.8804 | 0.5204 | Overfitting demonstration |
| **K-Nearest Neighbors** | 0.1310 | 0.5549 | 0.2119 | 0.7280 | 0.1772 | Suffers from sparse matrix |

### 💡 Why did Linear SVM outperform complex tree ensembles?
This is the most common viva question!
1. **High Dimensionality**: Our feature matrix has **10,810 dimensions**. In very high-dimensional sparse spaces (like TF-IDF text), data points are naturally linearly separable.
2. **Maximum Margin Principle**: SVM does not just find a boundary; it finds the boundary that maximizes the gap between real and fake postings.
3. **L2 Regularization**: SVM's built-in L2 regularization penalizes extreme weights, preventing it from overfitting to sparse outlier tokens where decision trees often fail.

---

## 8. Unsupervised Learning (SVD, K-Means, SOM)

In Unit 4 of the project, we tested unsupervised learning to see if clustering could detect fake jobs without human labels:

1. **TruncatedSVD (2D PCA equivalent)**: Projects the 10,810 features down to 2 principal components to visualize overall document topology.
2. **K-Means Clustering**: Uses Elbow curve (SSE inertia) and Silhouette score analysis across $k=2$ to $k=8$ to group jobs into distinct clusters.
3. **Self-Organizing Maps (MiniSom)**: An unsupervised artificial neural network that maps high-dimensional input onto a 2D topological grid.

### 🔑 Critical Theoretical Insight: Clustering vs Classification
> **Can unsupervised clustering replace supervised classification for fake job detection?**  
> **Answer: NO.**  
> Unsupervised clustering organizes job postings based on **job role and industry vocabulary** (e.g., Software Engineering vs Nursing vs Marketing). Fraudulent job postings deliberately imitate genuine listings in every industry. Therefore, fake jobs are dispersed across all occupational clusters rather than forming an isolated "fraud cluster". Supervised labels are mandatory to detect the subtle fraud markers.

---

## 9. Model Explainability & Structural Red Flags

A "black box" prediction that says "Fraud" without explanation is useless to recruiters and applicants. Our system provides multi-tier interpretability:

1. **Salient TF-IDF Tokens**: Shows which specific words in the job posting pushed the score towards fraud:
   - *Scam Indicators*: `"wire transfer"`, `"starter kit"`, `"make money"`, `"western union"`, `"clerk"`, `"data entry"`.
   - *Legitimate Indicators*: `"software engineer"`, `"tech company"`, `"health insurance"`, `"pto"`, `"bachelor degree"`.
2. **Structural Red Flags**:
   - Company logo missing.
   - Salary range withheld.
   - Personal free webmail addresses (`@gmail.com`, `@yahoo.com`) instead of verified corporate domains (`@company.com`).
   - Advance fee payment or wire transfer request detected.

---

## 10. Full-Stack Architecture & Code Flow

```
┌────────────────────────────────────────────────────────┐
│                   React 19 Frontend                    │
│            (Vite SPA · Light Theme · Port 5173)        │
│  Pages: Overview · Detect · Dataset · Models · History │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP POST /api/predict
                           ▼
┌────────────────────────────────────────────────────────┐
│                 Express Backend (Node.js)               │
│                       (Port 5001)                      │
│   - Validates incoming job posting payload             │
│   - Proxies inference to Python ML service             │
│   - Persists prediction & red flags into MongoDB       │
│   - Serves static charts and metrics from disk         │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP POST http://localhost:8000/predict
                           ▼
┌────────────────────────────────────────────────────────┐
│            FastAPI Python ML Inference Microservice    │
│                       (Port 8000)                      │
│   - Loads `ml/models/best_model.joblib` (SVM)          │
│   - Vectorizes text with fitted `tfidf.joblib`         │
│   - Encodes metadata with `ohe.joblib`                 │
│   - Returns: { label, fraud_probability, top_words,    │
│                red_flags }                             │
└────────────────────────────────────────────────────────┘
```

---

## 11. How to Run and Demo the Project

### One-Command Startup
In the root `fake-job-detector/` folder, run:
```bash
./start.sh
# or:
npm start
```

### Manual Individual Commands
```bash
# Terminal 1 — Python ML Service (FastAPI)
cd fake-job-detector
source ml/venv/bin/activate
uvicorn ml.service:app --port 8000 --host 0.0.0.0

# Terminal 2 — Express API Backend
cd fake-job-detector/server
npm start

# Terminal 3 — React Frontend (Vite)
cd fake-job-detector/client
npm run dev
```

### How to Demo to Professors in 2 Minutes:
1. Open **`http://localhost:5173`** in your browser.
2. Go to the **Detect** page (`/predict`).
3. Click the red button **"Load Fraudulent Sample"** $\rightarrow$ Click **"Analyze Job Posting"**.
   - Notice how the page smoothly scrolls up to reveal the result.
   - The card displays: **FRAUDULENT (100% Risk Score)**.
   - Red flags appear: *No company logo*, *Salary not disclosed*, *Contains scam keywords*.
   - Salient words appear: `"starter"`, `"wire"`, `"make money"`.
4. Click the green button **"Load Genuine Sample"** $\rightarrow$ Click **"Analyze Job Posting"**.
   - The card displays: **REAL (0% Risk Score)**.
   - Clean indicators: Verified logo, standard requirements.
5. Click **Models** (`/metrics`) to show the real evaluation table where **SVM** is highlighted as Champion.
6. Click **Insights** (`/insights`) to show the interactive ROC curve, PR curve, and Confusion Matrix.

---

## 12. College Viva Voce Top 12 Q&A Cheat Sheet

### Q1: What is the main objective of this project?
> **Answer**: To design and deploy an automated, explainable machine learning system capable of classifying online job postings as genuine or fraudulent using NLP text analysis and categorical metadata.

### Q2: Why did you not use Accuracy as the evaluation metric?
> **Answer**: Because of severe class imbalance (only 4.8% fraud). A naive model predicting "Real" for every single job achieves 95.2% accuracy while failing to identify a single scam. We used **Precision-Recall AUC (PR-AUC)** and **F1-Score**, which directly penalize false positives and false negatives on the minority class.

### Q3: What is Data Leakage and how did your pipeline prevent it?
> **Answer**: Data leakage happens when information from the test dataset leaks into the training pipeline before evaluation. We prevented it by performing the 80/20 Stratified Split first, fitting TF-IDF and One-Hot encoders only on the training set, and applying SMOTE strictly to training data.

### Q4: Which model gave the best result and why?
> **Answer**: Linear Support Vector Machine (LinearSVC with Platt calibration) achieved the highest PR-AUC of **0.9361** and an F1-Score of **0.8553**. In sparse, high-dimensional spaces (10,810 TF-IDF features), text is naturally linearly separable, and SVM’s maximum-margin hyperplane with L2 regularization avoids the overfitting that tree models experience.

### Q5: How did you handle the 95:5 class imbalance?
> **Answer**: We applied SMOTE (Synthetic Minority Over-sampling Technique) exclusively to the training set, synthesizing realistic fraudulent examples along the feature-space vectors of neighboring fraud instances to bring the training set to a 50:50 ratio.

### Q6: How does TF-IDF work in simple terms?
> **Answer**: Term Frequency measures how often a word appears in a specific job posting. Inverse Document Frequency downweights words that appear everywhere across the entire dataset (like "the", "work", "job"). High TF-IDF scores are given to terms that appear frequently in a specific posting but rarely across other listings (e.g., "wire transfer", "starter kit").

### Q7: Why did you demonstrate overfitting with Decision Trees?
> **Answer**: An unconstrained decision tree splits until every training sample is classified, achieving a 100% training score but dropping to 71% on test data because it memorized noise. By applying pruning parameters (`max_depth=10`, `min_samples_leaf=5`), we reduced the generalization gap and improved test robustness.

### Q8: What did you learn from the unsupervised clustering analysis?
> **Answer**: Unsupervised clustering (K-Means and SOM) groups postings by occupational domain (engineering vs administration vs medical), rather than fraud vs legitimate. Because scam postings deliberately mimic legitimate vacancies in every sector, supervised classification is mandatory.

### Q9: How is the system explainable?
> **Answer**: The system extracts the specific TF-IDF n-grams from the posting that received the highest model weights and surfaces heuristic red flags such as missing logos, withheld salaries, and personal webmail addresses.

### Q10: What is the purpose of the FastAPI Python microservice?
> **Answer**: To decouple the machine learning inference from the web server. The Python service loads the heavy scikit-learn models in memory once at startup and exposes a clean REST endpoint (`/predict`). The Express backend simply communicates via HTTP, keeping the Node.js process fast and lightweight.

### Q11: What database is used and what is stored?
> **Answer**: MongoDB is used to store:
> 1. Every prediction performed by users (audit log of inputs, probability, flags, timestamp).
> 2. Model benchmark metrics seeded from offline training runs.

### Q12: What is the Future Scope of this project?
> **Answer**:
> 1. Replacing n-gram TF-IDF with deep contextual transformer embeddings (**BERT/RoBERTa**).
> 2. Using **Graph Neural Networks (GNN)** to track relationships between recruiter accounts and detect organized scam syndicates.
> 3. Packaging the classifier into a **browser extension** for real-time verification directly on LinkedIn and Indeed.
