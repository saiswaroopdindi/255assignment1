# Part 2 · Replicating the `dlmastery/data_science_examples` experiments

> 🎥 **Video walkthrough:** to be uploaded

Five experiments from the professor's [data_science_examples](https://github.com/dlmastery/data_science_examples) repo,
re-implemented with an agentic coding assistant from the original prompts ([`PROMPTS.md`](PROMPTS.md)). Every project follows
CRISP-DM, includes an **AutoResearch hill-climbing loop** (every experiment logged, kept or rejected), uses leakage-safe
validation, and writes `outputs/RESULTS.md` with a plain-English paraphrase of the results.

| # | Project | Dataset (auto-downloaded) | Methods | Headline metric |
|---|---|---|---|---|
| 01 | [NYC Taxi trip duration](01_nyc_taxi/) | NYC TLC Yellow Taxi Jan-2024 (~3 M trips) + zone lookup | Greedy feature-group hill-climb, out-of-fold zone-pair target encoding, HPO; Ridge, RF, HistGB, XGBoost vs 2 baselines; ETA estimator | RMSLE on the unseen last week |
| 03 | [Customer segmentation](03_customer_segmentation/) | UCI/Kaggle Online Retail (541 k lines → ~4.3 k customers) | RFM + behaviour; K-Means, Ward, GMM, DBSCAN × k × feature sets; bootstrap ARI stability; PCA + t-SNE; named segments | Silhouette, Davies–Bouldin, ARI |
| 04 | [Market-basket mining](04_market_basket/) | UCI/Kaggle Online Retail (UK) | Apriori vs FP-Growth; support × confidence search; **rules validated on later months** (mine / validate / test by time); rule network; recommender | Validated rules, test-quarter survival |
| 06 | [Anomaly detection](06_anomaly_detection/) | Kaggle Credit-Card Fraud via OpenML 1597 (284 k txns, 0.17 % fraud) | Isolation Forest (hill-climbed), LOF, One-Class SVM, MCD, PCA & autoencoder reconstruction, rank ensemble vs supervised LR/GBM | PR-AUC, precision@k, recall@500 |
| 12 | [Time-series forecasting](12_timeseries_forecasting/) | UCI/Kaggle Bike Sharing (731 days) | STL, ADF/KPSS, 40-lag ACF/PACF; naive, seasonal-naive, MA, Holt-Winters, SARIMA, recursive GBM; 12-origin rolling backtest; forecast fan | MASE (vs seasonal naive) |

## Run

```bash
cd assignment1 && source .venv/bin/activate       # see ../README.md for setup
cd part2_ds_replication
python 12_timeseries_forecasting/timeseries_forecasting.py
python 03_customer_segmentation/customer_segmentation.py
python 04_market_basket/market_basket_mining.py
python 06_anomaly_detection/anomaly_detection.py
python 01_nyc_taxi/nyc_taxi_trip_duration.py
streamlit run dashboard.py                         # admin dashboard for all five projects
```
Each folder also has a `.ipynb` version. In Jupyter/Colab, start the kernel **inside that project folder**, because the notebooks import `../_common.py`.
Datasets are cached in `part2_ds_replication/data/` (git-ignored).

### Execution status

All five scripts and notebooks are present and pass Python syntax compilation. A local run was attempted for the time-series experiment, but the UCI download host was unavailable in the execution environment (`nodename nor servname provided`). Run the commands above on a network-enabled machine; do not present placeholder metrics as experimental results.

## Data-science audit checklist (applied to every project)
- [x] Test data touched exactly once; hyper-parameters / thresholds tuned on a separate validation slice
- [x] Temporal splits for time-ordered data (taxi, retail rules, fraud, bike demand) → no look-ahead
- [x] Encoders, scalers and target encodings fit on training data only (out-of-fold for training rows)
- [x] Metrics suited to the problem: RMSLE, PR-AUC instead of accuracy, MASE against a naive benchmark, hold-out rule survival
- [x] Baselines reported next to every model
- [x] Seeds fixed (`SEED = 42`) for reproducibility
- [x] Report sentences generated from computed numbers, not hard-coded claims

## Results
After running, see each project's `outputs/RESULTS.md` (or the dashboard). Video narration: [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md).
