# Prompts used — Part 1

The course prompt was reused from the professor's example and adapted from the *Adult income* dataset to the *Telco churn*
dataset. Because an agentic coding assistant (Claude in Cowork) was used instead of ChatGPT's code interpreter, I asked for a reproducible
notebook and script, not just a chat answer.

## 1. Custom instruction (persona)
> You are an industry expert and top notch data scientist. You are also expert in CRISP-DM methodology for data science and
> written several highly cited research papers of case studies of using crisp-dm data science for various data sets in kaggle.

## 2. Main prompt (adapted from the class example)
> As an industry expert data scientist and expert in CRISP-DM methodology for data science, given the Kaggle data set for
> **telco customer churn prediction** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), can you do step by step EDA analysis playing
> the role of an expert data scientist professor teaching a masters program in data science. Your response should be textbook
> quality. Make sure you prepare all the steps and reports of the methodology in principled fashion on data understanding, data
> analysis, data visualization, data cleaning, data pre-processing, feature selection, data clustering, outlier analysis and
> processing, **classification** for prediction of churn using a variety of methods compared to a baseline (including a
> **deep-learning model**), providing proper metrics for these techniques and a final recommendation. Use your most popular ML
> package like scikit-learn etc. Please respond in many reasonable small sized chunks starting with the initial steps of
> CRISP-DM methodology through all the steps on this data set. Repeat the requirement in each chunk towards the end of the
> chunk and explain where you are (mind-map) in the process and what you will do next. Assume you have limited compute and
> hence chunk the processing steps carefully.

## 3. Agentic additions (what I asked the coding agent on top)
1. "Make it a single reproducible script that also converts to a Jupyter notebook; save every figure and metric to `outputs/`."
2. "Guard against data leakage: split the test set first, keep all preprocessing in sklearn Pipelines, and choose the decision
   threshold on out-of-fold training predictions only."
3. "Add a business cost matrix (missed churner = $500, wasted offer = $100) and optimise the threshold for it."
4. "Add explainability (permutation importance, SHAP, logistic odds ratios), a model card, and a Streamlit app for deployment."
5. "Auto-generate the results report and a Medium article draft from the actual run so the numbers never drift."

## 4. Mind-map of the chunks
```
CRISP-DM
├── 1 Business understanding  → cost matrix, success criteria
├── 2 Data understanding      → quality report, target balance, EDA plots, chi²/Mann-Whitney tests
├── 3 Data preparation        → cleaning, feature engineering, IQR + IsolationForest outliers,
│                               K-Means segments, mutual-information feature ranking
├── 4 Modeling                → baseline + 9 models (5-fold CV), PyTorch DNN, randomized tuning
├── 5 Evaluation              → test metrics, cost-optimal threshold, ROC/PR, calibration, gains, explainability
└── 6 Deployment              → joblib pipeline, Streamlit app, model card, monitoring plan
```
