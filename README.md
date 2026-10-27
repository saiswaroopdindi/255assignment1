# CMPE 255 · Assignment 1 — AI-Assisted End-to-End Data Science

**Student:** Sai Swaroop Dindi · **Course:** CMPE 255 Data Mining, Fall 2026 · **AI assistant used:** Codex (agentic coding)

> Assignment showcase: open [`docs/index.html`](docs/index.html) locally, or publish the `docs/` folder with GitHub Pages.

| Deliverable | Link |
|---|---|
| 🎥 Part 1 video — Telco churn end-to-end | **To be recorded and uploaded** |
| 🎥 Part 2 video — Replicating the professor's experiments | **To be recorded and uploaded** |
| ✍️ Medium article (Part 1) | [`part1_telco_churn/medium_article.md`](part1_telco_churn/medium_article.md) |
| 💬 Chat transcript | [`transcript/`](transcript/) |

---

## Part 1 · Telco Customer Churn with CRISP-DM → [`part1_telco_churn/`](part1_telco_churn/)

The professor's CRISP-DM persona and chunked prompt, applied to the **Kaggle Telco Customer Churn** dataset. The pipeline covers:

- **Business understanding:** an explicit cost matrix.
- **Data understanding:** data-quality checks, EDA and statistical tests.
- **Data preparation:** feature engineering, outlier analysis, K-Means segmentation and mutual-information feature selection.
- **Modeling:** a majority-class baseline against 8 classifiers (logistic regression, decision tree, random forest, gradient boosting, SVM, k-NN, an MLP neural net and XGBoost) plus an optional **PyTorch deep network**, plus hyper-parameter tuning.
- **Evaluation:** a cost-optimal decision threshold, calibration, lift, permutation importance, SHAP and odds ratios.
- **Deployment:** a Streamlit app and a model card.

The script also generates the Medium article draft from the actual run. The completed run selected logistic regression with test ROC-AUC **0.846** and reduced the modeled test cost from **$187,000 to $59,600** using a 0.40 threshold.

📄 Prompts: [`part1_telco_churn/PROMPTS.md`](part1_telco_churn/PROMPTS.md) · 🎬 Video script: [`part1_telco_churn/VIDEO_SCRIPT.md`](part1_telco_churn/VIDEO_SCRIPT.md) · 📊 Results: `part1_telco_churn/outputs/RESULTS.md`

## Part 2 · Replicating `dlmastery/data_science_examples` → [`part2_ds_replication/`](part2_ds_replication/)

| # | Experiment | Highlights |
|---|---|---|
| 01 | NYC Taxi trip duration | 3 M TLC trips, temporal split, AutoResearch feature/HPO hill-climb, out-of-fold zone-pair encoding, ETA estimator |
| 03 | Customer segmentation | RFM, 4 algorithms × k × feature sets, bootstrap stability, t-SNE, named segments |
| 04 | Market-basket mining | Apriori vs FP-Growth, thresholds tuned on later months, rules tested on an unseen quarter, recommender |
| 06 | Anomaly detection | 6 unsupervised detectors + ensemble vs supervised, PR-AUC / precision@k, hill-climbed Isolation Forest |
| 12 | Time-series forecasting | STL, ADF/KPSS, ACF/PACF, 6 forecasters, 12-origin rolling backtest, forecast fan |

The Streamlit admin dashboard (`streamlit run part2_ds_replication/dashboard.py`) shows all five projects. The Part 2 scripts and notebooks are included; their dataset-backed execution is pending because this environment cannot resolve the UCI download host.
📄 Prompts: [`part2_ds_replication/PROMPTS.md`](part2_ds_replication/PROMPTS.md) · 🎬 Video script: [`part2_ds_replication/VIDEO_SCRIPT.md`](part2_ds_replication/VIDEO_SCRIPT.md)

---

## Reproduce everything

```bash
git clone <YOUR_REPO_URL> && cd <repo>/assignment1
bash run_all.sh            # creates .venv, installs requirements, builds notebooks, runs all 6 experiments (~30 min)
streamlit run part1_telco_churn/app.py
streamlit run part2_ds_replication/dashboard.py
```
Requirements: Python 3.10–3.12, about 3 GB free disk, internet access (the datasets download automatically).
On macOS, if XGBoost complains about `libomp`, run `brew install libomp` or ignore it, since XGBoost is optional everywhere.

## Repository layout
```
assignment1/
├── README.md                ← you are here
├── requirements.txt · run_all.sh
├── part1_telco_churn/       churn_crispdm.py/.ipynb, features.py, app.py, PROMPTS.md, VIDEO_SCRIPT.md,
│                            medium_article.md, outputs/{RESULTS.md, MODEL_CARD.md, metrics.json, figures/}
├── part2_ds_replication/    _common.py, dashboard.py, PROMPTS.md, VIDEO_SCRIPT.md,
│   ├── 01_nyc_taxi/                 nyc_taxi_trip_duration.py/.ipynb, outputs/
│   ├── 03_customer_segmentation/    customer_segmentation.py/.ipynb, outputs/
│   ├── 04_market_basket/            market_basket_mining.py/.ipynb, outputs/
│   ├── 06_anomaly_detection/        anomaly_detection.py/.ipynb, outputs/
│   └── 12_timeseries_forecasting/   timeseries_forecasting.py/.ipynb, outputs/
├── tools/py_to_ipynb.py     converts the scripts to notebooks
└── transcript/              exported AI chat transcript(s)
```

## The end-to-end AI-assisted journey
1. **Prompting:** I reused the professor's "industry expert CRISP-DM data scientist" persona and chunked, mind-mapped prompt, and added agentic requirements: reproducible scripts, leakage guards, business cost, auto-generated reports.
2. **Building:** the agent wrote each project as a single script that converts into a notebook, so the code you read is the code that runs.
3. **Auditing:** the runnable code was syntax-checked, and Part 1 was executed end to end. The default runner uses bounded, single-process tuning for laptop portability; the optional PyTorch comparison is opt-in with `RUN_TORCH=1`.
4. **Reporting:** Part 1 writes `RESULTS.md`, `MODEL_CARD.md`, metrics JSON, figures and a plain-English **paraphrase** of its own numbers. The Medium article and video script build on those. Part 2 will write the same outputs after its datasets are available.

## Lessons learned
- Ask the AI for a *process* (CRISP-DM, validation design), not only a model, and verify leakage yourself.
- Choose metrics for the decision: cost-weighted thresholds, PR-AUC for rare events, MASE against a naive forecast, hold-out rule survival.
- Generate reports from code so the write-up never drifts from the results.
## Additional project: Sentiment Analysis with Twitter Sentiment Dataset

This section preserves the documentation already present in the GitHub repository before the assignment files were added.

This repository contains a simple Python script to train and run a sentiment analysis model on the Kaggle/Twitter Sentiment dataset (training.1600000.processed.noemoticon.csv).

Dataset notes
- Format: CSV with 6 fields: polarity, id, date, query, user, text.
- Polarity: 0 = negative, 2 = neutral, 4 = positive.
- The training labels were heuristically created from emoticons: tweets with :) treated as positive, :( as negative.

Files
- `main.py`: trainer and predictor CLI.
- `requirements.txt`: Python dependencies.

Quickstart
1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```


2. Train a model:

- Sample 5% (quick):

```bash
python3 main.py --train --data archive-2/training.1600000.processed.noemoticon.csv --sample 0.05 --model-out models/sentiment.joblib
```

- Full dataset (may take long and require significant RAM/CPU):

```bash
python3 main.py --train --data archive-2/training.1600000.processed.noemoticon.csv --model-out models/sentiment.joblib
```

3. Predict:

```bash
python3 main.py --predict "I love this!" "This is terrible." --model models/sentiment.joblib
```

Acknowledgements
- Dataset inspired by the Kaggle Twitter Sentiment dataset.
