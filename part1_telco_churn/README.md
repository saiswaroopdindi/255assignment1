# Part 1 · Telco Customer Churn — End-to-End CRISP-DM

> 🎥 **Video walkthrough:** to be uploaded · ✍️ **Article draft:** [`medium_article.md`](medium_article.md)

An end-to-end data-science case study on the [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset (7,043 customers), built with an agentic AI coding assistant using the professor's *"expert CRISP-DM data scientist"*
persona and chunked prompt (see [`PROMPTS.md`](PROMPTS.md)).

## What's inside

| File | Purpose |
|---|---|
| [`churn_crispdm.py`](churn_crispdm.py) / [`churn_crispdm.ipynb`](churn_crispdm.ipynb) | The full CRISP-DM pipeline (same code; the notebook is generated from the script) |
| [`features.py`](features.py) | Leakage-free feature engineering shared by training and the app |
| [`app.py`](app.py) | Streamlit "what-if" churn-risk scorer + evaluation/EDA dashboard |
| [`PROMPTS.md`](PROMPTS.md) | Persona, main prompt, agentic follow-ups, CRISP-DM mind-map |
| [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) | Narration script for the YouTube walkthrough |
| `medium_article.md` | Medium article draft, **auto-generated with the real numbers** after a run |
| `outputs/RESULTS.md`, `outputs/MODEL_CARD.md`, `outputs/metrics.json` | Auto-generated report, model card and all metrics |
| `outputs/figures/*.png` | 13 figures (EDA, outliers, clusters, leaderboard, ROC/PR/calibration/gains, SHAP…) |

## CRISP-DM coverage

| Phase | What was done |
|---|---|
| 1 Business understanding | Churn framed as a cost problem (missed churner $500, wasted offer $100); success criteria defined |
| 2 Data understanding | Data-quality report, target balance, numeric/categorical EDA, χ² + Cramér's V, Mann-Whitney + rank-biserial, Spearman heatmap |
| 3 Data preparation | `TotalCharges` repair, 5 engineered features, stratified split **before** fitting, `ColumnTransformer` pipeline; IQR + Isolation-Forest outlier analysis; K-Means segmentation (silhouette-selected k); mutual-information feature ranking |
| 4 Modeling | Majority baseline + Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, SVM, k-NN, MLP neural net, XGBoost (5-fold CV) · optional **PyTorch DNN** · randomized hyper-parameter search |
| 5 Evaluation | Held-out test: ROC-AUC, PR-AUC, F1, recall, Brier; **cost-optimal threshold chosen on out-of-fold train predictions**; calibration curve; cumulative-gains/lift; permutation importance, SHAP, odds ratios |
| 6 Deployment | `joblib` pipeline, Streamlit app, model card, monitoring/retraining plan |

## Run it

```bash
cd assignment1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # + optional: pip install torch
cd part1_telco_churn
python churn_crispdm.py                  # ~5-10 min on a laptop; downloads the CSV automatically
streamlit run app.py                     # interactive app
```

**Google Colab:** upload this folder (or `git clone` the repo), then
`%cd assignment1/part1_telco_churn`, `!pip install -r ../requirements.txt` and run the notebook cells.
`features.py` must be in the same folder as the notebook.

If the automatic download fails, download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle into `part1_telco_churn/data/`.

## Results

After running, the numbers are in [`outputs/RESULTS.md`](outputs/RESULTS.md) (leaderboard, test metrics at the default and
cost-optimal thresholds, business-cost savings, lift, segments, top drivers).
