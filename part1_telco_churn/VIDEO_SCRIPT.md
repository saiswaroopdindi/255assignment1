# YouTube script — Part 1: Telco Churn end-to-end (target 8–10 min)

Fill the `[ ]` numbers from `outputs/RESULTS.md` after your run, and reword any sentence your results don't support (e.g. if the DNN wins). Record with QuickTime (File → New Screen Recording) or OBS.

| # | Time | On screen | What to say (paraphrase in your own words) |
|---|---|---|---|
| 1 | 0:00 | GitHub repo README | "Hi, I'm Sai Swaroop. This is CMPE 255 Assignment 1. Part 1 is an end-to-end churn-prediction project built with an AI coding agent using CRISP-DM." |
| 2 | 0:30 | `PROMPTS.md` | "I started from the professor's persona and chunked prompt, adapted it to the Telco dataset, and asked the agent for a reproducible notebook. I also asked it to guard against leakage and optimise for business cost." |
| 3 | 1:15 | Terminal: `python churn_crispdm.py` running | "One command downloads the data, runs all six CRISP-DM phases, and writes every figure and metric to `outputs/`." |
| 4 | 1:45 | `01_target_balance`, `03_categorical_churn_rates` | "About 26.5% of customers churn, so accuracy is misleading. Contract, online security, tech support, internet service and payment method show the strongest categorical associations." |
| 5 | 2:45 | `05_tenure_vs_charges`, `04_correlation_heatmap` | "Churners are new customers with high monthly bills. TotalCharges is almost tenure × monthly charge, so it's collinear." |
| 6 | 3:15 | `06_outlier_boxplots` + console | "Isolation Forest flagged 3.0% of rows as unusual, but they churn at 14.2% versus 26.5% overall, so I kept them instead of deleting them." |
| 7 | 3:45 | `07_kmeans_segments`, `cluster_profiles.csv` | "K-Means found 4 segments. The segment churn rates range from 14% to 49%, which helps target campaigns." |
| 8 | 4:15 | `08_mutual_information` | "Mutual information ranks contract, tenure and internet service as the most informative features." |
| 9 | 4:45 | `09_cv_leaderboard` | "I compared 9 models with 5-fold cross-validation against a majority-class baseline. Logistic regression leads with ROC-AUC 0.847; the MLP has nearly identical PR-AUC." |
| 10 | 5:30 | `10_pytorch_training_curve` | "The optional PyTorch comparison is disabled in the default portable run. The always-on sklearn MLP gives us the neural-network comparison without making the assignment depend on a slow local Torch build." |
| 11 | 6:00 | `11_evaluation_dashboard` | "On the held-out test set, ROC-AUC is 0.846 and PR-AUC is 0.653. I chose a 0.40 threshold on out-of-fold training predictions. Recall rises from 80.0% to 88.0%, and cost drops from $66,400 to $59,600, a 68.1% saving versus doing nothing. The top 20% captures 50.0% of churners." |
| 12 | 7:15 | `12_permutation_importance`, `14_shap_beeswarm` | "Permutation importance and SHAP agree with the EDA: contract, tenure, internet service and charges drive the predictions." |
| 13 | 7:45 | `streamlit run app.py` demo | "For deployment, the saved pipeline powers this app. The app accepts a customer profile, shows churn risk, and recommends an action at the selected threshold." |
| 14 | 8:30 | `medium_article.md` / Medium | "The Medium article was generated from the same run, so the numbers always match the code." |
| 15 | 9:00 | README | "Takeaways: prompt for a process, verify leakage yourself, and judge models by business cost, not accuracy. Thanks!" |
