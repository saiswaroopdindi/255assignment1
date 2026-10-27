# Telco Churn — Results Report (auto-generated)

**Data:** 7,043 customers, churn rate 26.5%. Train/test = 5,634/1,409 (stratified).

## Cross-validated model comparison (5-fold, train set)
| model                  |   roc_auc |   pr_auc |   recall |   precision |    f1 |   accuracy |
|:-----------------------|----------:|---------:|---------:|------------:|------:|-----------:|
| Logistic Regression    |     0.847 |    0.66  |    0.793 |       0.519 | 0.628 |      0.75  |
| Neural Net (MLP 64-32) |     0.845 |    0.661 |    0.524 |       0.668 | 0.585 |      0.803 |
| Random Forest          |     0.844 |    0.656 |    0.712 |       0.565 | 0.63  |      0.778 |
| Hist Gradient Boosting |     0.84  |    0.646 |    0.513 |       0.652 | 0.574 |      0.798 |
| XGBoost                |     0.838 |    0.648 |    0.506 |       0.655 | 0.571 |      0.798 |
| k-NN (k=25)            |     0.833 |    0.615 |    0.569 |       0.627 | 0.596 |      0.796 |
| SVM (RBF)              |     0.829 |    0.611 |    0.769 |       0.523 | 0.622 |      0.753 |
| Decision Tree          |     0.825 |    0.613 |    0.778 |       0.507 | 0.614 |      0.74  |
| Baseline (majority)    |     0.5   |    0.265 |    0     |       0     | 0     |      0.735 |

## Tuned models (CV ROC-AUC)
| model | CV ROC-AUC |
|---|---|
| Logistic Regression | 0.8479 |
| Hist Gradient Boosting | 0.8389 |
| XGBoost | 0.8479 |

PyTorch DNN: skipped by default for a portable one-command run (set RUN_TORCH=1 to enable); the sklearn MLP row above is the neural-network result.

## Final model: **Logistic Regression** — held-out test set
| metric | t = 0.50 | t = 0.37 (cost-optimal) |
|---|---|---|
| ROC-AUC | 0.845 | 0.845 |
| PR-AUC | 0.653 | 0.653 |
| Recall (churners caught) | 0.797 | 0.893 |
| Precision | 0.511 | 0.450 |
| F1 | 0.623 | 0.599 |
| Accuracy | 0.744 | 0.682 |
| Business cost | $66,500 | $60,800 |

* Do-nothing cost on the test set: **$187,000** → with model: **$60,800** (67.5% saving).
* Contacting the riskiest 20 % of customers reaches **50.0%** of all churners (lift 2.51×).
* Majority-class baseline ROC-AUC = 0.500, accuracy = 0.735 (shows why accuracy is misleading).

## Customer segments (K-Means, k=4)
|   cluster |   size |   churn_rate |   tenure |   monthly |   services |
|----------:|-------:|-------------:|---------:|----------:|-----------:|
|         0 |   1655 |         0.14 |    28.03 |     25.82 |       1.55 |
|         1 |    820 |         0.41 |    16.95 |     76.36 |       4.33 |
|         2 |   1322 |         0.49 |    12.95 |     73    |       4    |
|         3 |   1837 |         0.16 |    57.49 |     89.25 |       6.58 |

## Most important drivers (permutation importance, test set)
- **tenure**: ROC-AUC drop 0.0414
- **Contract**: ROC-AUC drop 0.0394
- **tenure_bucket**: ROC-AUC drop 0.0165
- **InternetService**: ROC-AUC drop 0.0140
- **OnlineSecurity**: ROC-AUC drop 0.0034
- **PaymentMethod**: ROC-AUC drop 0.0029
- **TechSupport**: ROC-AUC drop 0.0028
- **MultipleLines**: ROC-AUC drop 0.0028

## Outliers
Isolation Forest flagged 3.0% of training rows; their churn rate is
14.2% vs 26.5% overall, so they were kept as legitimate signal.

## Recommendation
1. Deploy **Logistic Regression** with decision threshold **0.37** (not 0.5): it trades extra
   cheap offers for many fewer missed churners.
2. Target the highest-risk segment first (see cluster table and top drivers: `tenure`, `Contract`, `tenure_bucket`, `InternetService`);
   typical levers are contract upgrades, bundled tech-support/security and auto-pay incentives.
3. Monitor drift monthly and re-train when PR-AUC degrades.
