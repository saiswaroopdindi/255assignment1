# YouTube script — Part 2: Replicating the professor's data-science experiments (target 10–12 min)

Show the Streamlit dashboard (`streamlit run dashboard.py`) for most of the video. Fill `[ ]` from each `outputs/RESULTS.md`,
and reword anything your run doesn't support.

| # | Time | On screen | What to say |
|---|---|---|---|
| 1 | 0:00 | Professor's repo → my `part2_ds_replication/` | "Part 2 replicates five experiments from the professor's repo using his original prompts with my AI coding agent: NYC taxi, clustering, market basket, anomaly detection and time series." |
| 2 | 0:40 | `PROMPTS.md` | "These are the verbatim prompts. I adapted them to reproducible scripts and asked for an autoresearch hill-climbing loop and a leakage audit in every project." |
| 3 | 1:20 | Dashboard → **01 NYC Taxi** | "Three million January-2024 TLC trips. Cleaning removed [x]% (zero-length, >3 h, impossible speeds). I split by time: the last week is the test set." |
| 4 | 2:00 | `03_hill_climb` | "The hill-climber adds one feature group at a time and keeps it only if validation RMSLE improves. Time of day and zone-pair encoding gave the biggest gains. Final validation RMSLE [x]." |
| 5 | 2:40 | `04_evaluation`, ETA table | "On the unseen week, [best] gets RMSLE [x] versus [y] for the distance-only baseline, about [z] minutes median error. The ETA examples show JFK to Midtown at rush hour taking [x] minutes." |
| 6 | 3:30 | **03 Segmentation** → `02_model_selection_curves` | "From online-retail transactions I built RFM features for [4,3xx] customers. I searched four algorithms, k from 2 to 10, and two feature sets. [Algo] with k=[k] gave silhouette [x], and bootstrap ARI [x] says the segments are stable." |
| 7 | 4:20 | `03_cluster_projections`, `04_segment_profiles` | "Paraphrase: segment [Champions] is only [x]% of customers but [y]% of revenue. Hibernating customers are many but low value." |
| 8 | 5:00 | **04 Market basket** → `01_eda` | "Same retailer, UK invoices. I mine rules on Dec–Jun, tune thresholds on Jul–Sep, and test on the holiday quarter Oct–Dec." |
| 9 | 5:40 | `02_rules_quality`, `03_rule_network` | "The best setting is support [x], confidence [y], giving [n] validated rules. [x]% of them still hold in the unseen quarter. Apriori and FP-Growth return identical itemsets, and FP-Growth is [x]× faster. Top rule: [rule], lift [x]." |
| 10 | 6:40 | **06 Anomaly** → `01_eda` | "Credit-card fraud is 0.17% of transactions, so I use PR-AUC, not accuracy. The split is by time." |
| 11 | 7:20 | `03_hill_climb`, `02_evaluation` | "I hill-climbed Isolation Forest on validation PR-AUC, then compared LOF, One-Class SVM, robust covariance, PCA and an autoencoder. The best unsupervised method is [x] with PR-AUC [y]. With labels, gradient boosting reaches [z]. ROC-AUC looks high for all of them, which is why it's misleading here." |
| 12 | 8:30 | **12 Time series** → `01_stl_decomposition`, `02_acf_pacf` | "Daily bike rentals: growth trend, yearly and weekly seasonality. ADF says it's non-stationary until differenced. The ACF/PACF show weekly spikes at lag 7." |
| 13 | 9:20 | `04_backtest`, `05_forecast_fan` | "Twelve rolling origins, 14-day horizon. [Best] has MASE [x] versus [y] for seasonal naive. The fan uses real backtest errors, and it's wide because weather isn't known two weeks ahead." |
| 14 | 10:20 | README audit checklist | "Across all five: temporal splits, out-of-fold encodings, test sets used once, baselines everywhere. An independent AI reviewer audited the code and I fixed every issue it found." |
| 15 | 11:00 | — | "Thanks for watching. Links are in the README." |
