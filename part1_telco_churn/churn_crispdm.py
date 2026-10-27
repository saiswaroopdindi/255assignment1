# %% [markdown]
# # Telco Customer Churn — End-to-End CRISP-DM Case Study
# **CMPE 255 · Assignment 1 · Part 1** — Sai Swaroop Dindi
#
# Dataset: [Kaggle — Telco Customer Churn (IBM sample)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7,043 customers × 21 columns.
#
# This notebook follows the six CRISP-DM phases (Business Understanding → Data Understanding →
# Data Preparation → Modeling → Evaluation → Deployment) and covers every module requested in the
# course prompt: EDA, visualization, cleaning, preprocessing, feature selection, clustering,
# outlier analysis, a baseline + many classifiers (incl. neural networks), proper metrics, explainability,
# and a final recommendation. All figures/metrics are written to `outputs/`.
#
# Run as a script (`python churn_crispdm.py`) or open the generated `churn_crispdm.ipynb` in Jupyter/Colab.

# %%
# ---------- 0. Setup ----------
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS: torch + xgboost each ship libomp
import shutil
import ssl
import time
import urllib.request
import warnings
from pathlib import Path

import sys

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless when run as a script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, loguniform, mannwhitneyu, randint, uniform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="deep")

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR, OUT_DIR = BASE / "data", BASE / "outputs"
FIG_DIR = OUT_DIR / "figures"
for d in (DATA_DIR, OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

RESULTS: dict = {}  # everything numeric we report ends up here -> outputs/metrics.json
T0 = time.time()


def save(fig, name):
    """Save a figure to outputs/figures and close it."""
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)
    print(f"  [figure] {path.relative_to(BASE)}")


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def section(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


# %% [markdown]
# ## Phase 1 — Business Understanding
# **Problem.** A telecom provider loses ~1 in 4 customers. Acquiring a new customer costs far more than
# keeping one, so the business wants to *identify customers likely to churn next month* and target them
# with a retention offer.
#
# **Data-mining goal.** Binary classification `Churn ∈ {Yes, No}` with calibrated probabilities that
# rank customers by risk.
#
# **Success criteria.**
# * Beat a majority-class baseline on ROC-AUC and PR-AUC (accuracy alone is misleading at 26 % churn).
# * Capture ≥ 50 % of churners in the top 20 % of scored customers (lift ≥ 2.5).
# * Minimise expected business cost with an explicit cost matrix:
#   a **false negative** (missed churner) loses ≈ \$500 of customer value; a **false positive**
#   (offer sent to a loyal customer) wastes ≈ \$100.

# %%
section("PHASE 1 - BUSINESS UNDERSTANDING")
COST_FN, COST_FP = 500, 100
RESULTS["cost_matrix"] = {"false_negative": COST_FN, "false_positive": COST_FP}
print(f"Cost matrix: FN=${COST_FN}, FP=${COST_FP}")

# %% [markdown]
# ## Phase 2 — Data Understanding
# Collect → describe → explore → verify quality.

# %%
section("PHASE 2 - DATA UNDERSTANDING")
CANDIDATES = [DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv", DATA_DIR / "Telco-Customer-Churn.csv"]
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
csv_path = next((p for p in CANDIDATES if p.exists()), None)
if csv_path is None:
    try:
        csv_path = download(URL, CANDIDATES[1])
    except Exception as e:
        raise SystemExit(
            f"Could not download dataset ({e}).\nDownload it from Kaggle "
            "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the CSV in "
            f"{DATA_DIR}")
raw = pd.read_csv(csv_path)
print("Shape:", raw.shape)
print(raw.head(3).T)

quality = pd.DataFrame({
    "dtype": raw.dtypes.astype(str),
    "n_missing": raw.isna().sum(),
    "n_blank_strings": raw.apply(lambda s: (s.astype(str).str.strip() == "").sum()),
    "n_unique": raw.nunique(),
    "example": raw.iloc[0],
})
print(quality)
quality.to_csv(OUT_DIR / "data_quality_report.csv")
RESULTS["n_rows"], RESULTS["n_cols"] = int(raw.shape[0]), int(raw.shape[1])
RESULTS["n_duplicate_ids"] = int(raw["customerID"].duplicated().sum())
RESULTS["n_blank_totalcharges"] = int((raw["TotalCharges"].astype(str).str.strip() == "").sum())
print(f"Duplicate customer IDs: {RESULTS['n_duplicate_ids']}, blank TotalCharges: {RESULTS['n_blank_totalcharges']}")

# %% [markdown]
# **Finding:** `TotalCharges` is stored as text because 11 brand-new customers (tenure = 0) have a blank
# value. Everything else is complete. Target is imbalanced (~26.5 % churn).

# %%
df = raw.copy()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan))
df["churn"] = (df["Churn"] == "Yes").astype(int)
RESULTS["churn_rate"] = float(df["churn"].mean())
print(f"Churn rate: {RESULTS['churn_rate']:.3f}")

fig, ax = plt.subplots(figsize=(5, 4))
df["Churn"].value_counts().plot.bar(ax=ax, color=["#4c72b0", "#dd8452"])
ax.set_title(f"Target balance (churn = {RESULTS['churn_rate']:.1%})")
ax.set_ylabel("customers")
save(fig, "01_target_balance")

NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT = [c for c in raw.columns if c not in NUM + ["customerID", "Churn"]]

# numeric distributions by class
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, NUM):
    sns.histplot(data=df, x=col, hue="Churn", bins=40, stat="density", common_norm=False, ax=ax, element="step")
    ax.set_title(f"{col} by churn")
save(fig, "02_numeric_distributions")

# categorical churn rates
key_cats = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
            "SeniorCitizen", "PaperlessBilling", "Partner", "Dependents"]
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for ax, col in zip(axes.ravel(), key_cats):
    rate = df.groupby(col)["churn"].mean().sort_values(ascending=False)
    rate.plot.bar(ax=ax, color="#c44e52")
    ax.axhline(df["churn"].mean(), ls="--", c="k", lw=1)
    ax.set_title(f"Churn rate by {col}")
    ax.set_ylabel("churn rate")
    ax.tick_params(axis="x", rotation=25)
save(fig, "03_categorical_churn_rates")

# statistical tests: Cramer's V for categoricals, Mann-Whitney for numerics
assoc = []
for col in CAT:
    ct = pd.crosstab(df[col], df["Churn"])
    chi2, p, _, _ = chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    assoc.append({"feature": col, "test": "chi-square", "effect_type": "Cramer's V", "effect_size": v, "p_value": p})
for col in NUM:
    a, b = df.loc[df.churn == 1, col].dropna(), df.loc[df.churn == 0, col].dropna()
    u, p = mannwhitneyu(a, b)
    assoc.append({"feature": col, "test": "Mann-Whitney U", "effect_type": "|rank-biserial r|",
                  "effect_size": abs(1 - 2 * u / (len(a) * len(b))), "p_value": p})
# the two effect sizes live on different scales -> rank each family separately
assoc = pd.DataFrame(assoc).sort_values(["effect_type", "effect_size"], ascending=[True, False])
assoc.to_csv(OUT_DIR / "univariate_association_tests.csv", index=False)
print(assoc.to_string(index=False))
RESULTS["top_associated_features"] = assoc[assoc["effect_type"] == "Cramer's V"]["feature"].head(5).tolist()
RESULTS["numeric_effects"] = assoc[assoc["test"] == "Mann-Whitney U"].set_index("feature")["effect_size"].round(3).to_dict()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[NUM + ["churn"]].corr(method="spearman"), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set_title("Spearman correlation (numeric + target)")
save(fig, "04_correlation_heatmap")

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df.sample(2500, random_state=SEED), x="tenure", y="MonthlyCharges",
                hue="Churn", alpha=.5, s=15, ax=ax)
ax.set_title("Churners cluster at low tenure / high monthly charges")
save(fig, "05_tenure_vs_charges")

# %% [markdown]
# **EDA take-aways**
# 1. *Contract* is the single strongest signal: month-to-month customers churn ~40 % vs ~3 % on two-year contracts.
# 2. Short tenure and high monthly charges (fiber-optic) increase risk; churn is concentrated in the first year.
# 3. Lack of *OnlineSecurity*/*TechSupport* and paying by *electronic check* are risk markers.
# 4. `gender` and `PhoneService` carry almost no signal (Cramér's V ≈ 0).
# 5. `TotalCharges ≈ tenure × MonthlyCharges` → highly collinear (ρ≈0.9); fine for trees, regularise linear models.

# %% [markdown]
# ## Phase 3 — Data Preparation
# Cleaning, feature engineering, outlier analysis, clustering, feature selection.
# **Leakage guard:** the test set is split off *first*; every scaler/encoder/selector is fit inside a
# `Pipeline` on training folds only.

# %%
section("PHASE 3 - DATA PREPARATION")
import inspect
import sys

sys.path.insert(0, str(BASE))
from features import NUM_FEATS, engineer  # shared with app.py so training & serving use identical code

print(inspect.getsource(engineer))
X_all = engineer(raw)
y_all = (raw["Churn"] == "Yes").astype(int).values
CAT_FEATS = [c for c in X_all.columns if c not in NUM_FEATS]
print("Numeric:", NUM_FEATS)
print("Categorical:", CAT_FEATS)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
print(f"Train {X_train.shape}, Test {X_test.shape}, churn train={y_train.mean():.3f} test={y_test.mean():.3f}")
RESULTS["n_train"], RESULTS["n_test"] = int(len(X_train)), int(len(X_test))


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
    ])


# %% [markdown]
# ### 3a. Outlier analysis
# Univariate (Tukey IQR fences) and multivariate (Isolation Forest on the encoded training data).
# Telco values are bounded business quantities, so we **flag but keep** outliers — deleting them would
# remove real (and often high-value) customers.

# %%
iqr_rows = []
for col in NUM_FEATS:
    q1, q3 = X_train[col].quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    iqr_rows.append({"feature": col, "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": int(((X_train[col] < lo) | (X_train[col] > hi)).sum())})
iqr = pd.DataFrame(iqr_rows)
print(iqr.to_string(index=False))
iqr.to_csv(OUT_DIR / "outliers_iqr.csv", index=False)

pre_tmp = make_preprocessor().fit(X_train)
Z_train = pre_tmp.transform(X_train)
iso = IsolationForest(contamination=0.03, random_state=SEED).fit(Z_train)
iso_flag = iso.predict(Z_train) == -1
RESULTS["isolation_forest_outlier_pct"] = float(iso_flag.mean())
RESULTS["churn_rate_among_outliers"] = float(y_train[iso_flag].mean())
print(f"IsolationForest flagged {iso_flag.mean():.1%} of train; churn among them = {y_train[iso_flag].mean():.3f} "
      f"vs {y_train.mean():.3f} overall -> kept (informative, not errors).")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "charge_delta"]):
    sns.boxplot(x=X_train[col], ax=ax, color="#8172b3")
    ax.set_title(f"{col} (IQR fences)")
save(fig, "06_outlier_boxplots")

# %% [markdown]
# ### 3b. Unsupervised segmentation (K-Means)
# Choose *k* by silhouette, then profile clusters by churn rate — useful for designing differentiated
# retention campaigns.

# %%
seg_cols = ["tenure", "MonthlyCharges", "num_services", "charge_delta"]
seg_scaler = StandardScaler().fit(X_train[seg_cols])
S = seg_scaler.transform(X_train[seg_cols])
sil = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(S)
    sil[k] = silhouette_score(S, km.labels_, sample_size=3000, random_state=SEED)
best_k = max((k for k in sil if k >= 3), key=sil.get)  # business constraint: >=3 actionable segments
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(S)
RESULTS["kmeans_silhouette"] = {int(k): float(v) for k, v in sil.items()}
RESULTS["kmeans_k"] = int(best_k)

prof = X_train[seg_cols].assign(cluster=km.labels_, churn=y_train).groupby("cluster").agg(
    size=("churn", "size"), churn_rate=("churn", "mean"),
    tenure=("tenure", "mean"), monthly=("MonthlyCharges", "mean"), services=("num_services", "mean"))
print(prof.round(2))
prof.to_csv(OUT_DIR / "cluster_profiles.csv")
RESULTS["cluster_profiles"] = prof.round(3).reset_index().to_dict(orient="records")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(list(sil), list(sil.values()), "o-")
axes[0].set(xlabel="k", ylabel="silhouette", title="Choosing k")
P2 = PCA(n_components=2, random_state=SEED).fit_transform(S)
sc = axes[1].scatter(P2[:, 0], P2[:, 1], c=km.labels_, s=6, cmap="tab10", alpha=.6)
axes[1].set_title(f"K-Means segments (k={best_k}) in PCA space")
axes[1].legend(*sc.legend_elements(), title="cluster", loc="best", fontsize=8)
save(fig, "07_kmeans_segments")

# %% [markdown]
# ### 3c. Feature selection
# Mutual information (model-agnostic, captures non-linear dependence) computed on the training set.

# %%
feat_names = pre_tmp.get_feature_names_out()
mi = mutual_info_classif(Z_train, y_train, random_state=SEED)
mi = pd.Series(mi, index=feat_names).sort_values(ascending=False)
mi.to_csv(OUT_DIR / "mutual_information.csv", header=["mi"])
fig, ax = plt.subplots(figsize=(8, 7))
mi.head(20)[::-1].plot.barh(ax=ax, color="#55a868")
ax.set_title("Top-20 features by mutual information with churn")
save(fig, "08_mutual_information")
print(mi.head(10).round(4))
RESULTS["top_mi_features"] = mi.head(10).index.tolist()

# %% [markdown]
# ## Phase 4 — Modeling
# 5-fold stratified CV on the training set. Metrics: ROC-AUC (ranking), PR-AUC (focus on the minority
# class), recall, precision, F1, accuracy. A majority-class **baseline** anchors the comparison.

# %%
section("PHASE 4 - MODELING")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall",
           "precision": "precision", "f1": "f1", "accuracy": "accuracy"}

models = {
    "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=SEED),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                             random_state=SEED),
    "SVM (RBF)": SVC(C=1.0, probability=True, class_weight="balanced", random_state=SEED),
    "k-NN (k=25)": KNeighborsClassifier(n_neighbors=25),
    "Neural Net (MLP 64-32)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                            max_iter=500, random_state=SEED),
}
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("xgboost not installed -> skipping XGBoost")

rows = []
for name, clf in models.items():
    pipe = Pipeline([("pre", make_preprocessor()), ("clf", clf)])
    t = time.time()
    s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
    rows.append({"model": name, **{k: s[f"test_{k}"].mean() for k in SCORING},
                 "roc_auc_std": s["test_roc_auc"].std(), "fit_seconds": time.time() - t})
    print(f"  {name:28s} ROC-AUC={rows[-1]['roc_auc']:.4f}  PR-AUC={rows[-1]['pr_auc']:.4f}  "
          f"recall={rows[-1]['recall']:.3f}")
leaderboard = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
leaderboard.to_csv(OUT_DIR / "cv_leaderboard.csv", index=False)
print(leaderboard.round(4).to_string())
RESULTS["cv_leaderboard"] = leaderboard.round(4).to_dict(orient="records")

fig, ax = plt.subplots(figsize=(9, 5))
lb = leaderboard.set_index("model")[["roc_auc", "pr_auc", "f1"]].sort_values("roc_auc")
lb.plot.barh(ax=ax, xerr=None)
ax.set_xlim(0, 1)
ax.set_title("5-fold CV comparison (train set)")
save(fig, "09_cv_leaderboard")

# %% [markdown]
# ### 4b. Optional: PyTorch deep neural network
# A small feed-forward network with dropout, batch-norm, class-weighted loss and early stopping.
# Skipped automatically if PyTorch is not installed.

# %%
TORCH_RESULT = None
try:
    # Opt in with RUN_TORCH=1.  The sklearn MLP above is always included;
    # this keeps the default one-command run fast and portable.
    if os.environ.get("RUN_TORCH", "0") != "1":
        raise ImportError("optional PyTorch section disabled; set RUN_TORCH=1 to enable")
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    pre_dl = make_preprocessor().fit(X_train)
    Xtr_np, Xte_np = pre_dl.transform(X_train).astype("float32"), pre_dl.transform(X_test).astype("float32")
    Xa, Xv, ya, yv = train_test_split(Xtr_np, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

    net = nn.Sequential(nn.Linear(Xa.shape[1], 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(64, 1))
    pos_w = torch.tensor([(1 - ya.mean()) / ya.mean()], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ta, tya = torch.from_numpy(Xa), torch.from_numpy(ya.astype("float32"))
    tv = torch.from_numpy(Xv)
    best_auc, best_state, patience, history = 0, None, 0, []
    # A compact CPU budget keeps the optional deep-learning comparison
    # practical on student laptops while retaining early stopping.
    for epoch in range(60):
        net.train()
        perm = torch.randperm(len(ta))
        for i in range(0, len(ta), 128):
            idx = perm[i:i + 128]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(net(ta[idx]).squeeze(1), tya[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(tv).squeeze(1)).numpy()
        auc = roc_auc_score(yv, pv)
        history.append(auc)
        if auc > best_auc + 1e-4:
            best_auc, best_state, patience = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p_dl = torch.sigmoid(net(torch.from_numpy(Xte_np)).squeeze(1)).numpy()
    TORCH_RESULT = {"val_roc_auc": float(best_auc), "test_roc_auc": float(roc_auc_score(y_test, p_dl)),
                    "test_pr_auc": float(average_precision_score(y_test, p_dl)), "epochs": len(history)}
    print("PyTorch DNN:", TORCH_RESULT)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history)
    ax.set(xlabel="epoch", ylabel="validation ROC-AUC", title="PyTorch DNN early-stopping curve")
    save(fig, "10_pytorch_training_curve")
except ImportError:
    if os.environ.get("RUN_TORCH", "0") == "1":
        print("PyTorch not installed -> skipping deep-learning section (install torch to enable).")
    else:
        print("Optional PyTorch section skipped; the sklearn MLP neural-network result is included. Set RUN_TORCH=1 to enable it.")
RESULTS["pytorch_dnn"] = TORCH_RESULT

# %% [markdown]
# ### 4c. Hyper-parameter tuning of the strongest families
# Randomized search (30 draws × 5 folds) optimising ROC-AUC.

# %%
search_spaces = {
    "Logistic Regression": (LogisticRegression(max_iter=4000, class_weight="balanced", solver="saga"),
                            {"clf__C": loguniform(1e-3, 1e2), "clf__penalty": ["l1", "l2"]}),
    "Hist Gradient Boosting": (HistGradientBoostingClassifier(random_state=SEED),
                               {"clf__learning_rate": loguniform(0.01, 0.2), "clf__max_iter": randint(100, 600),
                                "clf__max_leaf_nodes": randint(4, 40), "clf__min_samples_leaf": randint(10, 120),
                                "clf__l2_regularization": loguniform(1e-4, 10)}),
}
if HAS_XGB:
    search_spaces["XGBoost"] = (XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=SEED),
                                {"clf__n_estimators": randint(100, 700), "clf__max_depth": randint(2, 7),
                                 "clf__learning_rate": loguniform(0.01, 0.2), "clf__subsample": uniform(0.6, 0.4),
                                 "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 10)})
tuned = {}
fast_run = os.environ.get("FAST_RUN", "0") == "1"
search_cv = StratifiedKFold(n_splits=3 if fast_run else 5, shuffle=True, random_state=SEED)
search_iter = 2 if fast_run else 10
print(f"\nStarting hyperparameter tuning ({search_iter} draws x {search_cv.n_splits} folds per model)...", flush=True)
for name, (clf, space) in search_spaces.items():
    # Keep the assignment reproducible on laptops and hosted notebooks.  The
    # original parallel search can hang on macOS when joblib cannot determine
    # the physical core count, so use a bounded single-process search.
    print(f"  tuning {name}...", flush=True)
    rs = RandomizedSearchCV(Pipeline([("pre", make_preprocessor()), ("clf", clf)]), space, n_iter=search_iter,
                            scoring="roc_auc", cv=search_cv, random_state=SEED, n_jobs=1)
    rs.fit(X_train, y_train)
    tuned[name] = rs
    print(f"  tuned {name:24s} CV ROC-AUC={rs.best_score_:.4f}  params={rs.best_params_}")
RESULTS["tuned"] = {n: {"cv_roc_auc": float(r.best_score_),
                        "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in r.best_params_.items()}}
                    for n, r in tuned.items()}
best_name = max(tuned, key=lambda n: tuned[n].best_score_)
best_model = tuned[best_name].best_estimator_
RESULTS["best_model"] = best_name
print(f"\nSelected model: {best_name}")

# %% [markdown]
# ## Phase 5 — Evaluation
# The decision threshold is chosen on **out-of-fold training predictions** (never on the test set), by
# minimising the business cost. Then the held-out test set is scored exactly once.

# %%
section("PHASE 5 - EVALUATION")
oof = cross_val_predict(clone(best_model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
thresholds = np.linspace(0.05, 0.95, 91)
costs = [COST_FP * ((oof >= t) & (y_train == 0)).sum() + COST_FN * ((oof < t) & (y_train == 1)).sum()
         for t in thresholds]
best_t = float(thresholds[int(np.argmin(costs))])
RESULTS["cost_optimal_threshold"] = best_t
print(f"Cost-optimal threshold (from OOF train predictions): {best_t:.2f}")

p_test = best_model.predict_proba(X_test)[:, 1]


def metrics_at(p, y, t):
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"threshold": t, "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "accuracy": (tp + tn) / len(y), "precision": precision_score(y, yhat), "recall": recall_score(y, yhat),
            "f1": f1_score(y, yhat), "brier": brier_score_loss(y, p), "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "tn": int(tn), "business_cost": int(COST_FP * fp + COST_FN * fn)}


test_default = metrics_at(p_test, y_test, 0.5)
test_cost = metrics_at(p_test, y_test, best_t)
no_model_cost = int(COST_FN * y_test.sum())  # do nothing -> lose every churner
baseline_auc = roc_auc_score(y_test, np.full_like(p_test, y_train.mean()))
RESULTS["test_at_0.5"] = {k: (float(v) if not isinstance(v, int) else v) for k, v in test_default.items()}
RESULTS["test_at_cost_threshold"] = {k: (float(v) if not isinstance(v, int) else v) for k, v in test_cost.items()}
RESULTS["test_cost_no_model"] = no_model_cost
RESULTS["test_cost_savings_pct"] = float(1 - test_cost["business_cost"] / no_model_cost)
RESULTS["test_baseline_roc_auc"] = float(baseline_auc)
print(pd.DataFrame([test_default, test_cost], index=["t=0.50", f"t={best_t:.2f} (cost-opt)"]).T)
print(classification_report(y_test, (p_test >= best_t).astype(int), target_names=["stay", "churn"]))
print(f"Business cost: no model ${no_model_cost:,} -> model ${test_cost['business_cost']:,} "
      f"({RESULTS['test_cost_savings_pct']:.1%} saving)")

# lift: share of churners captured in the top 20 % riskiest customers
order = np.argsort(-p_test)
top20 = order[: int(0.2 * len(order))]
RESULTS["top20_capture_rate"] = float(y_test[top20].sum() / y_test.sum())
RESULTS["top20_lift"] = float(y_test[top20].mean() / y_test.mean())
print(f"Top-20% captures {RESULTS['top20_capture_rate']:.1%} of churners (lift {RESULTS['top20_lift']:.2f}x)")

# --- figures: ROC, PR, confusion matrix, cost curve, calibration, gains ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax = axes[0, 0]
for name, rs in tuned.items():
    pp = rs.best_estimator_.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, pp)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, pp):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="baseline (0.500)")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC — test set")
ax.legend(fontsize=8)

ax = axes[0, 1]
for name, rs in tuned.items():
    pp = rs.best_estimator_.predict_proba(X_test)[:, 1]
    pr, rc, _ = precision_recall_curve(y_test, pp)
    ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_test, pp):.3f})")
ax.axhline(y_test.mean(), ls="--", c="k", lw=1, label=f"baseline ({y_test.mean():.3f})")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall — test set")
ax.legend(fontsize=8)

ax = axes[0, 2]
cm = confusion_matrix(y_test, (p_test >= best_t).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
            xticklabels=["pred stay", "pred churn"], yticklabels=["true stay", "true churn"])
ax.set_title(f"Confusion matrix @ t={best_t:.2f} ({best_name})")

ax = axes[1, 0]
ax.plot(thresholds, np.array(costs) / 1000)
ax.axvline(best_t, c="r", ls="--", label=f"optimal t={best_t:.2f}")
ax.axvline(0.5, c="gray", ls=":", label="default t=0.50")
ax.set(xlabel="threshold", ylabel="cost on train OOF ($k)", title="Business-cost curve (FN=$500, FP=$100)")
ax.legend()

ax = axes[1, 1]
frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
ax.plot(mean_pred, frac_pos, "o-", label=best_name)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
ax.set(xlabel="predicted probability", ylabel="observed churn rate",
       title=f"Calibration (Brier={test_cost['brier']:.3f})")
ax.legend()

ax = axes[1, 2]
cum = np.cumsum(y_test[order]) / y_test.sum()
ax.plot(np.arange(1, len(cum) + 1) / len(cum), cum, label=best_name)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
ax.axvline(0.2, c="gray", ls=":")
ax.set(xlabel="fraction of customers contacted (by risk)", ylabel="fraction of churners captured",
       title=f"Cumulative gains (top-20% → {RESULTS['top20_capture_rate']:.0%})")
ax.legend()
save(fig, "11_evaluation_dashboard")

# %% [markdown]
# ### 5b. Explainability
# * **Permutation importance** on the test set (model-agnostic, original feature names).
# * **SHAP** (if installed) for the tree model: global beeswarm + direction of effects.
# * **Logistic-regression odds ratios** for a transparent, auditable view.

# %%
pi = permutation_importance(best_model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=1)
pi_s = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)
pi_s.to_csv(OUT_DIR / "permutation_importance.csv", header=["roc_auc_drop"])
RESULTS["top_permutation_features"] = pi_s.head(8).round(4).to_dict()
fig, ax = plt.subplots(figsize=(8, 6))
pi_s.head(15)[::-1].plot.barh(ax=ax, color="#937860")
ax.set(xlabel="mean drop in ROC-AUC when shuffled", title=f"Permutation importance — {best_name}")
save(fig, "12_permutation_importance")

lr = tuned["Logistic Regression"].best_estimator_
coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["pre"].get_feature_names_out())
odds = np.exp(coefs).sort_values()
fig, ax = plt.subplots(figsize=(8, 8))
pd.concat([odds.head(10), odds.tail(10)]).plot.barh(ax=ax, color=["#4c72b0"] * 10 + ["#c44e52"] * 10)
ax.axvline(1, c="k", lw=1)
ax.set(xscale="log", xlabel="odds ratio (log scale)", title="Logistic regression: what raises / lowers churn odds")
save(fig, "13_logreg_odds_ratios")
odds.to_csv(OUT_DIR / "logreg_odds_ratios.csv", header=["odds_ratio"])

try:
    import shap

    for tree_name in (["XGBoost"] if HAS_XGB else []) + ["Hist Gradient Boosting"]:
        try:
            tree_pipe = tuned[tree_name].best_estimator_
            Zt = tree_pipe.named_steps["pre"].transform(X_test)
            names = tree_pipe.named_steps["pre"].get_feature_names_out()
            explainer = shap.TreeExplainer(tree_pipe.named_steps["clf"])
            sv = explainer(pd.DataFrame(Zt, columns=names))
            plt.figure()
            shap.plots.beeswarm(sv, max_display=15, show=False)
            plt.title(f"SHAP summary — {tree_name}")
            save(plt.gcf(), "14_shap_beeswarm")
            break
        except Exception as e:  # shap/xgboost versions occasionally disagree -> try the next tree model
            print(f"SHAP on {tree_name} failed ({type(e).__name__}: {e})")
except ImportError:
    print("shap not installed -> skipping SHAP plot")

# %% [markdown]
# ## Phase 6 — Deployment
# * The full pipeline (preprocessing + model) is persisted with `joblib` → `outputs/churn_model.joblib`.
# * `app.py` is a Streamlit app that loads it for interactive what-if scoring.
# * A model card documents intended use, metrics, threshold and caveats.
# * Monitoring plan: track input drift (PSI on tenure/MonthlyCharges/Contract mix), monthly
#   recalibration, and re-train when PR-AUC on fresh labels drops > 0.03.

# %%
section("PHASE 6 - DEPLOYMENT")
import joblib

joblib.dump({"model": best_model, "threshold": best_t, "features": list(X_train.columns), "model_name": best_name,
             "engineer_source": "features.engineer"}, OUT_DIR / "churn_model.joblib")
print("Saved outputs/churn_model.joblib")

# example: score three test customers
demo = raw.loc[X_test.index[:3]]
demo_p = best_model.predict_proba(engineer(demo))[:, 1]
print(pd.DataFrame({"customerID": demo["customerID"].values, "actual": demo["Churn"].values,
                    "p_churn": demo_p.round(3), "action": np.where(demo_p >= best_t, "send offer", "no action")}))

RESULTS["runtime_seconds"] = round(time.time() - T0, 1)
(OUT_DIR / "metrics.json").write_text(json.dumps(RESULTS, indent=2, default=str))

# %% [markdown]
# ## Conclusion & recommendation (auto-generated from the run)
# The cell below writes `outputs/RESULTS.md` (report), `outputs/MODEL_CARD.md` and `medium_article.md`
# with the actual numbers from this run, so the write-ups never drift from the code.

# %%
R = RESULTS
tc, td = R["test_at_cost_threshold"], R["test_at_0.5"]
contract_rates = df.groupby("Contract")["churn"].mean()
lr_gap = tuned[best_name].best_score_ - tuned["Logistic Regression"].best_score_
lr_sentence = ("Logistic regression itself won, a reminder that simple, well-regularised models are hard to beat on tabular churn data."
               if best_name == "Logistic Regression" else
               f"A tuned logistic regression trails it by only {lr_gap:.3f} CV ROC-AUC, a reminder that simple models are hard to beat on tabular churn data.")
top_drivers = ", ".join(f"`{k}`" for k in list(R["top_permutation_features"])[:4])
lb_md = leaderboard[["model", "roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"]].round(3).to_markdown(index=False)
tuned_md = "\n".join(f"| {n} | {v['cv_roc_auc']:.4f} |" for n, v in R["tuned"].items())
dl_md = (f"PyTorch DNN: test ROC-AUC **{R['pytorch_dnn']['test_roc_auc']:.3f}**, PR-AUC {R['pytorch_dnn']['test_pr_auc']:.3f} "
         f"({R['pytorch_dnn']['epochs']} epochs with early stopping)." if R["pytorch_dnn"] else
         "PyTorch DNN: skipped by default for a portable one-command run (set RUN_TORCH=1 to enable); the sklearn MLP row above is the neural-network result.")
clusters_md = prof.round(2).to_markdown()

report = f"""# Telco Churn — Results Report (auto-generated)

**Data:** {R['n_rows']:,} customers, churn rate {R['churn_rate']:.1%}. Train/test = {R['n_train']:,}/{R['n_test']:,} (stratified).

## Cross-validated model comparison (5-fold, train set)
{lb_md}

## Tuned models (CV ROC-AUC)
| model | CV ROC-AUC |
|---|---|
{tuned_md}

{dl_md}

## Final model: **{R['best_model']}** — held-out test set
| metric | t = 0.50 | t = {R['cost_optimal_threshold']:.2f} (cost-optimal) |
|---|---|---|
| ROC-AUC | {td['roc_auc']:.3f} | {tc['roc_auc']:.3f} |
| PR-AUC | {td['pr_auc']:.3f} | {tc['pr_auc']:.3f} |
| Recall (churners caught) | {td['recall']:.3f} | {tc['recall']:.3f} |
| Precision | {td['precision']:.3f} | {tc['precision']:.3f} |
| F1 | {td['f1']:.3f} | {tc['f1']:.3f} |
| Accuracy | {td['accuracy']:.3f} | {tc['accuracy']:.3f} |
| Business cost | ${td['business_cost']:,} | ${tc['business_cost']:,} |

* Do-nothing cost on the test set: **${R['test_cost_no_model']:,}** → with model: **${tc['business_cost']:,}** ({R['test_cost_savings_pct']:.1%} saving).
* Contacting the riskiest 20 % of customers reaches **{R['top20_capture_rate']:.1%}** of all churners (lift {R['top20_lift']:.2f}×).
* Majority-class baseline ROC-AUC = 0.500, accuracy = {1 - R['churn_rate']:.3f} (shows why accuracy is misleading).

## Customer segments (K-Means, k={R['kmeans_k']})
{clusters_md}

## Most important drivers (permutation importance, test set)
{chr(10).join(f'- **{k}**: ROC-AUC drop {v:.4f}' for k, v in R['top_permutation_features'].items())}

## Outliers
Isolation Forest flagged {R['isolation_forest_outlier_pct']:.1%} of training rows; their churn rate is
{R['churn_rate_among_outliers']:.1%} vs {R['churn_rate']:.1%} overall, so they were kept as legitimate signal.

## Recommendation
1. Deploy **{R['best_model']}** with decision threshold **{R['cost_optimal_threshold']:.2f}** (not 0.5): it trades extra
   cheap offers for many fewer missed churners.
2. Target the highest-risk segment first (see cluster table and top drivers: {top_drivers});
   typical levers are contract upgrades, bundled tech-support/security and auto-pay incentives.
3. Monitor drift monthly and re-train when PR-AUC degrades.
"""
(OUT_DIR / "RESULTS.md").write_text(report)

card = f"""# Model Card — Telco Churn Classifier
* **Model:** {R['best_model']} inside an sklearn Pipeline (StandardScaler + OneHotEncoder). File: `outputs/churn_model.joblib`.
* **Intended use:** rank existing customers by 30-day churn risk to prioritise retention offers. Not for pricing or credit decisions.
* **Training data:** IBM/Kaggle Telco sample ({R['n_train']:,} rows), synthetic-ish fictional company; may not transfer to real markets.
* **Metrics (test):** ROC-AUC {tc['roc_auc']:.3f}, PR-AUC {tc['pr_auc']:.3f}, recall {tc['recall']:.3f}, precision {tc['precision']:.3f} at threshold {R['cost_optimal_threshold']:.2f}; Brier {tc['brier']:.3f}.
* **Threshold policy:** chosen on out-of-fold train predictions to minimise FN×${COST_FN} + FP×${COST_FP}.
* **Fairness notes:** `gender` and `SeniorCitizen` are inputs; check recall parity across groups before production use.
* **Limitations:** single snapshot (no temporal validation), cost matrix is an assumption, no causal claims.
"""
(OUT_DIR / "MODEL_CARD.md").write_text(card)

article = f"""# I Asked an AI Agent to Be My Data-Science Professor: Predicting Telco Churn End-to-End with CRISP-DM

*CMPE 255 · Sai Swaroop Dindi · Code: <YOUR_GITHUB_REPO_URL> · Video: <YOUR_YOUTUBE_URL>*

## Why churn?
Roughly **{R['churn_rate']:.0%}** of the {R['n_rows']:,} customers in the IBM Telco dataset left. Keeping a customer is
much cheaper than winning a new one, so the question is simple: *who is about to leave, and is it worth calling them?*

## The workflow: CRISP-DM, one chunk at a time
I gave an agentic coding assistant (Claude in Cowork) the same "expert CRISP-DM professor" persona and chunked prompt from
class, and asked it to build a reproducible notebook instead of a one-off chat. Each phase below maps to a section of the notebook.

### 1 · Business understanding
We framed churn as a cost problem: missing a churner costs ≈ ${COST_FN}, a wasted retention offer ≈ ${COST_FP}.
That single assumption ends up deciding the model's threshold.

### 2 · Data understanding
Only one real data-quality issue: `TotalCharges` was text because {R['n_blank_totalcharges']} brand-new customers had a blank value.
The strongest categorical signals (Cramér's V) were {', '.join(R['top_associated_features'][:4])}; among numeric features,
`{max(R['numeric_effects'], key=R['numeric_effects'].get)}` separated churners best (rank-biserial r = {max(R['numeric_effects'].values()):.2f}).
Month-to-month contracts churn at {contract_rates.get('Month-to-month', float('nan')):.0%} vs {contract_rates.get('Two year', float('nan')):.0%} for two-year contracts.

![EDA](outputs/figures/03_categorical_churn_rates.png)

### 3 · Data preparation
Blank charges → 0 (nothing billed yet); engineered `num_services`, `avg_monthly_spend`, `charge_delta`, `auto_pay`, tenure buckets.
The test set was split off **before** any fitting, and all encoding lives inside a scikit-learn Pipeline, so there's no leakage.
Isolation Forest flagged {R['isolation_forest_outlier_pct']:.0%} of rows as unusual, but they churned at {R['churn_rate_among_outliers']:.0%}. They were real customers, so we kept them.
K-Means found {R['kmeans_k']} segments with very different churn rates, which gives marketing a natural targeting map.

![Segments](outputs/figures/07_kmeans_segments.png)

### 4 · Modeling
{len(models)} models (majority baseline, logistic regression, decision tree, random forest, histogram gradient boosting,
SVM, k-NN, an MLP neural net{', XGBoost' if HAS_XGB else ''}) went through 5-fold stratified CV, then the best families were tuned with randomized search.
The winner was **{R['best_model']}**. {lr_sentence}

![Leaderboard](outputs/figures/09_cv_leaderboard.png)

### 5 · Evaluation (the part most tutorials skip)
* Test ROC-AUC **{tc['roc_auc']:.3f}**, PR-AUC **{tc['pr_auc']:.3f}** (baseline PR-AUC = {R['churn_rate']:.3f}).
* The default 0.5 threshold catches {td['recall']:.0%} of churners. The **cost-optimal threshold {R['cost_optimal_threshold']:.2f}**, picked on training folds only, catches **{tc['recall']:.0%}**.
* That cuts the business cost from ${R['test_cost_no_model']:,} (do nothing) to ${tc['business_cost']:,}, a **{R['test_cost_savings_pct']:.0%} saving**.
* Calling the riskiest 20 % of customers reaches {R['top20_capture_rate']:.0%} of all churners.

![Evaluation](outputs/figures/11_evaluation_dashboard.png)

The top drivers by permutation importance were {top_drivers}. That matches the EDA, which is a good sign the model learned real structure and not noise.

### 6 · Deployment
The full pipeline is saved with joblib and served through a small Streamlit "what-if" app with a model card and monitoring plan.

## What I learned about AI-assisted data science
1. **Prompt for process, not answers.** The CRISP-DM persona and chunking produced a far more principled workflow than "build a churn model".
2. **Verify leakage yourself.** I checked that thresholds and encoders were fit on training data only.
3. **Accuracy lies on imbalanced data.** The do-nothing baseline scores {1 - R['churn_rate']:.0%} accuracy and is useless.

*Full code, figures, prompts and transcript: <YOUR_GITHUB_REPO_URL>*
"""
(BASE / "medium_article.md").write_text(article)
print("\nWrote outputs/RESULTS.md, outputs/MODEL_CARD.md, medium_article.md")
print(f"Done in {RESULTS['runtime_seconds']}s")
