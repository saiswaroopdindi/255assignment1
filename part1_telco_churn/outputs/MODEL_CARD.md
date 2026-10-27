# Model Card — Telco Churn Classifier
* **Model:** Logistic Regression inside an sklearn Pipeline (StandardScaler + OneHotEncoder). File: `outputs/churn_model.joblib`.
* **Intended use:** rank existing customers by 30-day churn risk to prioritise retention offers. Not for pricing or credit decisions.
* **Training data:** IBM/Kaggle Telco sample (5,634 rows), synthetic-ish fictional company; may not transfer to real markets.
* **Metrics (test):** ROC-AUC 0.845, PR-AUC 0.653, recall 0.893, precision 0.450 at threshold 0.37; Brier 0.166.
* **Threshold policy:** chosen on out-of-fold train predictions to minimise FN×$500 + FP×$100.
* **Fairness notes:** `gender` and `SeniorCitizen` are inputs; check recall parity across groups before production use.
* **Limitations:** single snapshot (no temporal validation), cost matrix is an assumption, no causal claims.
