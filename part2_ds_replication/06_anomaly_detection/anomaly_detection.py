# %% [markdown]
# # 06 · Anomaly Detection — Credit-Card Fraud (unsupervised vs semi-supervised vs supervised)
# Replication of `dlmastery/data_science_examples/06_anomaly_detection`.
#
# **Original prompt:** *"Now lets do another project - anomaly detection using popular kaggle data set and popular methods -
# make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers
# and implement autoresearch to do hill climbing … include all details a data scientist and ai engineer will care."*
#
# **Data:** Kaggle *Credit Card Fraud Detection* (ULB, 284,807 transactions, 492 frauds = 0.17 %), fetched from OpenML
# (id 1597, identical file, no Kaggle login needed). **Papers:** Liu et al. 2008 (Isolation Forest), Breunig et al. 2000
# (LOF), Schölkopf et al. 2001 (One-Class SVM), Rousseeuw 1999 (MCD / robust covariance), autoencoder reconstruction error.

# %%
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_here.parent))
from _common import DATA, Project, plt  # noqa: E402

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

SEED = 42
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid")
P = Project(_here, "06 · Anomaly Detection (credit-card fraud)")

# %% [markdown]
# ## 1 · Business understanding
# Fraud analysts can review only a small number of alerts per day. So the model must **rank** transactions well:
# primary metric = **PR-AUC** (average precision), plus **precision@k** and **recall@k** for an alert budget k.
# ROC-AUC is reported but is overly optimistic at 0.17 % prevalence.

# %%
P.section("2 · Data understanding")
cc = fetch_openml(data_id=1597, as_frame=True, parser="auto", data_home=str(DATA / "openml"))
X = cc.data.astype(float)
y = (pd.Series(cc.target).astype(str).str.strip().str.replace("'", "") == "1").astype(int).values
print(X.shape, f"fraud rate = {y.mean():.4%}")
P.results.update({"n": int(len(y)), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean())})

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
axes[0].bar(["legit", "fraud"], np.bincount(y), color=["#4c72b0", "#c44e52"])
axes[0].set(yscale="log", title="Extreme class imbalance (log scale)")
sns.histplot(x=np.log1p(X["Amount"]), hue=y, stat="density", common_norm=False, bins=60, ax=axes[1], element="step")
axes[1].set(title="log(1+Amount) by class", xlabel="log1p(Amount)")
hours = (X["Time"] // 3600) % 24
pd.DataFrame({"hour": hours, "y": y}).groupby("hour")["y"].mean().plot(ax=axes[2], marker="o", color="#c44e52")
axes[2].set(title="Fraud rate by hour (Time = seconds since first txn)", ylabel="fraud rate", xlabel="hour offset mod 24")
P.save(fig, "01_eda")

# %% [markdown]
# ## 3 · Data preparation
# * **Temporal split** on `Time` (first 60 % train · next 15 % validation · last 25 % test) — mimics production, no look-ahead.
# * `Amount` → log1p; robust scaling (fit on train only). `V1…V28` are already PCA components.
# * Semi-supervised detectors are trained on **legitimate training transactions only** (novelty detection).

# %%
P.section("3 · Data preparation")
order = np.argsort(X["Time"].values, kind="stable")
X, y = X.iloc[order].reset_index(drop=True), y[order]
X["Amount"] = np.log1p(X["Amount"])
X["hour"] = (X["Time"] // 3600) % 24
X = X.drop(columns=["Time"])
n = len(X)
i_tr, i_va = int(0.60 * n), int(0.75 * n)
Xtr, Xva, Xte = X.iloc[:i_tr], X.iloc[i_tr:i_va], X.iloc[i_va:]
ytr, yva, yte = y[:i_tr], y[i_tr:i_va], y[i_va:]
scaler = RobustScaler().fit(Xtr)
Ztr, Zva, Zte = (scaler.transform(a) for a in (Xtr, Xva, Xte))
Zn = Ztr[ytr == 0]  # normal-only training data
sub = Zn[rng.choice(len(Zn), size=min(30_000, len(Zn)), replace=False)]
print(f"train {len(ytr):,} ({ytr.sum()} fraud) | val {len(yva):,} ({yva.sum()}) | test {len(yte):,} ({yte.sum()})")
P.results["split"] = {"train": [len(ytr), int(ytr.sum())], "val": [len(yva), int(yva.sum())], "test": [len(yte), int(yte.sum())]}

# %% [markdown]
# ## 4 · Modeling
# ### 4a · AutoResearch hill-climb for Isolation Forest (validation PR-AUC)

# %%
P.section("4 · Modeling")
log = []
inc = {"n_estimators": 200, "max_samples": 256, "max_features": 1.0}
best_ap = average_precision_score(yva, -IsolationForest(random_state=SEED, n_jobs=-1, **inc).fit(Zn).score_samples(Zva))
log.append({"step": 0, **inc, "val_pr_auc": best_ap, "kept": True})
for step in range(1, 13):
    prop = dict(inc)
    knob = rng.choice(["n_estimators", "max_samples", "max_features"])
    if knob == "n_estimators":
        prop[knob] = int(rng.choice([100, 200, 400, 600]))
    elif knob == "max_samples":
        prop[knob] = int(rng.choice([128, 256, 512, 1024, 4096]))
    else:
        prop[knob] = float(rng.choice([0.3, 0.5, 0.7, 1.0]))
    ap = average_precision_score(yva, -IsolationForest(random_state=SEED, n_jobs=-1, **prop).fit(Zn).score_samples(Zva))
    kept = ap > best_ap + 1e-4
    log.append({"step": step, **prop, "val_pr_auc": ap, "kept": kept})
    print(f"step {step:2d} {prop} val PR-AUC={ap:.4f} {'KEEP' if kept else 'reject'}")
    if kept:
        inc, best_ap = prop, ap
hill = P.table(pd.DataFrame(log), "autoresearch_isolation_forest")
P.results["iforest_best_params"], P.results["iforest_val_pr_auc"] = inc, float(best_ap)

# %% [markdown]
# ### 4b · Detector zoo (scores: higher = more anomalous)

# %%
scores_va, scores_te, fit_time = {}, {}, {}


def run(name, fit_fn):
    t = time.time()
    s_va, s_te = fit_fn()
    fit_time[name] = time.time() - t
    scores_va[name], scores_te[name] = s_va, s_te
    print(f"  {name:32s} val PR-AUC={average_precision_score(yva, s_va):.4f} ({fit_time[name]:.1f}s)")


iso = IsolationForest(random_state=SEED, n_jobs=-1, **inc).fit(Zn)
run("Isolation Forest (tuned)", lambda: (-iso.score_samples(Zva), -iso.score_samples(Zte)))
lof = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1).fit(sub)
run("Local Outlier Factor", lambda: (-lof.score_samples(Zva), -lof.score_samples(Zte)))
oc_sub = sub[:8000]
oc_scaler = StandardScaler().fit(oc_sub)
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(oc_scaler.transform(oc_sub))
run("One-Class SVM (rbf)", lambda: (-ocsvm.decision_function(oc_scaler.transform(Zva)),
                                    -ocsvm.decision_function(oc_scaler.transform(Zte))))
ee = EllipticEnvelope(contamination=0.01, support_fraction=0.9, random_state=SEED).fit(sub)
run("Robust covariance (MCD)", lambda: (-ee.score_samples(Zva), -ee.score_samples(Zte)))
pca = PCA(n_components=12, random_state=SEED).fit(Zn)


def pca_err(Z):
    return ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).sum(axis=1)


run("PCA reconstruction error", lambda: (pca_err(Zva), pca_err(Zte)))
ae_scaler = StandardScaler().fit(sub)
ae = MLPRegressor(hidden_layer_sizes=(20, 8, 20), activation="relu", max_iter=60, early_stopping=True,
                  random_state=SEED).fit(ae_scaler.transform(sub), ae_scaler.transform(sub))


def ae_err(Z):
    A = ae_scaler.transform(Z)
    return ((A - ae.predict(A)) ** 2).sum(axis=1)


run("Autoencoder (MLP 20-8-20)", lambda: (ae_err(Zva), ae_err(Zte)))

# unsupervised ensemble: average of normalised ranks of the detectors
uns = list(scores_va)
scores_va["Rank-average ensemble (unsup.)"] = np.mean([rankdata(scores_va[k]) for k in uns], axis=0)
scores_te["Rank-average ensemble (unsup.)"] = np.mean([rankdata(scores_te[k]) for k in uns], axis=0)

# supervised references (use labels) - upper bound on what labels buy you
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ztr, ytr)
run("Supervised: Logistic Regression", lambda: (lr.predict_proba(Zva)[:, 1], lr.predict_proba(Zte)[:, 1]))
hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, class_weight="balanced",
                                     random_state=SEED).fit(Ztr, ytr)
run("Supervised: Gradient Boosting", lambda: (hgb.predict_proba(Zva)[:, 1], hgb.predict_proba(Zte)[:, 1]))

# %% [markdown]
# ## 5 · Evaluation on the untouched last 25 % of time

# %%
P.section("5 · Evaluation")
K = int(yte.sum())  # alert budget = number of true frauds in the test period
rows = []
for name, s in scores_te.items():
    top = np.argsort(-s)[:K]
    top500 = np.argsort(-s)[:500]
    rows.append({"detector": name, "type": "supervised" if name.startswith("Supervised") else "unsupervised",
                 "test_PR_AUC": average_precision_score(yte, s), "test_ROC_AUC": roc_auc_score(yte, s),
                 f"precision@{K}": yte[top].mean(), "recall@500_alerts": yte[top500].sum() / yte.sum(),
                 "fit_seconds": fit_time.get(name, np.nan)})
board = P.table(pd.DataFrame(rows).sort_values("test_PR_AUC", ascending=False), "test_leaderboard")
print(board.round(4).to_string(index=False))
P.results["test_leaderboard"] = board.round(4).to_dict(orient="records")
P.results["alert_budget_k"] = K

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
for name, s in scores_te.items():
    pr, rc, _ = precision_recall_curve(yte, s)
    axes[0].plot(rc, pr, lw=1.3, ls="--" if name.startswith("Supervised") else "-",
                 label=f"{name} ({average_precision_score(yte, s):.3f})")
axes[0].axhline(yte.mean(), c="k", lw=.8)
axes[0].set(xlabel="recall", ylabel="precision", title="Precision–Recall (test)", yscale="log")
axes[0].legend(fontsize=7)
board.set_index("detector")["test_PR_AUC"].sort_values().plot.barh(ax=axes[1], color="#8172b3")
axes[1].set(title="Test PR-AUC by detector", xlabel="average precision")
best_uns = board[board["type"] == "unsupervised"].iloc[0]["detector"]
s = scores_te[best_uns]
axes[2].hist(s[yte == 0], bins=80, alpha=.6, density=True, label="legit")
axes[2].hist(s[yte == 1], bins=80, alpha=.6, density=True, label="fraud")
axes[2].set(title=f"Score separation — {best_uns}", xlabel="anomaly score")
axes[2].legend()
P.save(fig, "02_evaluation")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(hill["step"], hill["val_pr_auc"], "o", c="lightgray", label="proposal")
ax.plot(hill["step"], hill["val_pr_auc"].where(hill["kept"]).cummax().ffill(), c="#c44e52", lw=2, label="incumbent")
ax.set(xlabel="experiment", ylabel="validation PR-AUC", title="Isolation Forest hill-climb")
ax.legend()
P.save(fig, "03_hill_climb")

R = P.results
best_sup = board[board["type"] == "supervised"].iloc[0]
bu = board[board["detector"] == best_uns].iloc[0]
P.finish(f"""# 06 · Anomaly Detection — Results

* {R['n']:,} transactions, {R['n_fraud']} frauds ({R['fraud_rate']:.3%}). Temporal split 60/15/25 %.
* Isolation-Forest hill-climb ({len(hill)} experiments) → `{R['iforest_best_params']}`, validation PR-AUC {R['iforest_val_pr_auc']:.3f}.
* Alert budget k = {R['alert_budget_k']} (number of frauds in the test period).

## Test leaderboard
{board.round(4).to_markdown(index=False)}

**Paraphrase:** without any labels, the best detector (**{best_uns}**) reaches PR-AUC {bu['test_PR_AUC']:.3f}, about
{bu['test_PR_AUC'] / yte.mean():.0f}× better than random alerting. With 500 alerts it catches {bu['recall@500_alerts']:.0%} of frauds.
When labels exist, **{best_sup['detector']}** reaches PR-AUC {best_sup['test_PR_AUC']:.3f}. The practical
recommendation is to use unsupervised scores to surface *new* fraud patterns, and a supervised model, re-trained as analyst
labels arrive, for known patterns. Across detectors ROC-AUC spans {board['test_ROC_AUC'].min():.3f}–{board['test_ROC_AUC'].max():.3f},
while PR-AUC spans {board['test_PR_AUC'].min():.3f}–{board['test_PR_AUC'].max():.3f}. ROC-AUC hides the differences that matter at
{R['fraud_rate']:.2%} prevalence, which is why PR-AUC is the headline metric.
""")
