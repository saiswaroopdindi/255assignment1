# %% [markdown]
# # 03 · Customer Segmentation with Clustering (CRISP-DM + AutoResearch)
# Replication of `dlmastery/data_science_examples/03_customer_segmentation_clustering`.
#
# **Original prompt:** *"Now lets do another project - clustering using popular kaggle data set - make sure you follow
# crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement
# autoresearch to do hill climbing … include all details a data scientist and ai engineer will care."*
#
# **Data:** UCI / Kaggle *Online Retail* — 541,909 transactions of a UK gift retailer (Dec-2010 → Dec-2011).
# Customers are described with **RFM** (Recency, Frequency, Monetary) — the classic marketing-segmentation framework
# (Hughes 1994) — plus behavioural features.

# %%
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_here.parent))
from _common import Project, load_online_retail, plt  # noqa: E402

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
sns.set_theme(style="whitegrid")
P = Project(_here, "03 · Customer Segmentation (RFM clustering)")

# %% [markdown]
# ## 1 · Business understanding
# Marketing wants a small number of **actionable, stable** customer segments (e.g. *Champions*, *At-risk*) to tailor
# campaigns. Success = well-separated clusters (silhouette, Davies–Bouldin), stable under resampling (ARI), and
# segments that differ meaningfully in revenue.

# %%
P.section("2 · Data understanding")
raw = load_online_retail()
print(raw.shape)
print(raw.head())
q = {"missing CustomerID": raw["CustomerID"].isna().mean(),
     "cancellations (Invoice starts with C)": raw["InvoiceNo"].astype(str).str.startswith("C").mean(),
     "quantity <= 0": (raw["Quantity"] <= 0).mean(), "unit price <= 0": (raw["UnitPrice"] <= 0).mean(),
     "duplicate rows": raw.duplicated().mean()}
quality = P.table(pd.DataFrame({"check": q.keys(), "share_of_rows": q.values()}), "data_quality")
print(quality.round(4).to_string(index=False))
P.results["raw_rows"] = int(len(raw))

# %% [markdown]
# ## 3 · Data preparation — customer-level RFM table
# Drop anonymous rows, cancellations, non-positive quantities/prices and exact duplicates; aggregate to one row per customer.

# %%
P.section("3 · Data preparation")
df = raw.dropna(subset=["CustomerID"]).drop_duplicates()
df = df[~df["InvoiceNo"].astype(str).str.startswith("C") & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
cust = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum"),
    Tenure=("InvoiceDate", lambda s: (snapshot - s.min()).days),
    UniqueProducts=("StockCode", "nunique"),
)
cust["AvgBasket"] = cust["Monetary"] / cust["Frequency"]
P.results["n_customers"] = int(len(cust))
print(cust.describe().round(1))

FEATURE_SETS = {"RFM": ["Recency", "Frequency", "Monetary"],
                "RFM+behaviour": ["Recency", "Frequency", "Monetary", "Tenure", "UniqueProducts", "AvgBasket"]}
# heavy right skew -> log1p, then standardise (distance-based algorithms need comparable scales)
logged = np.log1p(cust)
scaled = {name: StandardScaler().fit_transform(logged[cols]) for name, cols in FEATURE_SETS.items()}

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes[0], ["Recency", "Frequency", "Monetary"]):
    sns.histplot(cust[col], bins=60, ax=ax, color="#4c72b0")
    ax.set_title(f"{col} (raw, skewed)")
for ax, col in zip(axes[1], ["Recency", "Frequency", "Monetary"]):
    sns.histplot(logged[col], bins=60, ax=ax, color="#55a868")
    ax.set_title(f"log1p({col})")
P.save(fig, "01_rfm_distributions")

# outliers: extreme wholesale buyers dominate k-means centroids if not handled -> flag with robust z-score on log scale
z = (logged - logged.median()) / (1.4826 * (logged - logged.median()).abs().median())
outlier = (z[["Frequency", "Monetary"]].abs() > 3.5).any(axis=1)
P.results["outlier_customers"] = int(outlier.sum())
print(f"Robust-z outliers: {outlier.sum()} customers ({outlier.mean():.1%}) -> kept (log scale tames them); reported separately")

# %% [markdown]
# ## 4 · Modeling — AutoResearch over algorithms × k × feature sets
# Search space: K-Means, Agglomerative (Ward), Gaussian Mixture, DBSCAN. Objective = silhouette, with
# Davies–Bouldin and Calinski–Harabasz logged. To avoid the "k=2 always wins" trap, the business constraint is k ≥ 3.

# %%
P.section("4 · AutoResearch")
log = []


def score(X, labels):
    mask = labels >= 0
    if len(set(labels[mask])) < 2:
        return None
    return {"silhouette": silhouette_score(X[mask], labels[mask], sample_size=min(3000, mask.sum()), random_state=SEED),
            "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
            "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
            "noise_share": float((~mask).mean())}


for fs, X in scaled.items():
    for k in range(2, 11):
        algos = {"KMeans": KMeans(n_clusters=k, n_init=10, random_state=SEED),
                 "Agglomerative-Ward": AgglomerativeClustering(n_clusters=k, linkage="ward"),
                 "GaussianMixture": GaussianMixture(n_components=k, covariance_type="full", random_state=SEED)}
        for a, model in algos.items():
            labels = model.fit_predict(X)
            s = score(X, labels)
            entry = {"feature_set": fs, "algorithm": a, "k": k, **s}
            if a == "KMeans":
                entry["inertia"] = model.inertia_
            if a == "GaussianMixture":
                entry["bic"] = model.bic(X)
            log.append(entry)
    # DBSCAN: eps from the k-distance "knee" (95th percentile of 2*dim-NN distance), a few multipliers
    d = np.sort(NearestNeighbors(n_neighbors=2 * X.shape[1]).fit(X).kneighbors(X)[0][:, -1])
    for mult in (0.6, 0.8, 1.0, 1.2):
        eps = float(np.quantile(d, 0.95) * mult)
        labels = DBSCAN(eps=eps, min_samples=2 * X.shape[1]).fit_predict(X)
        s = score(X, labels)
        if s:
            log.append({"feature_set": fs, "algorithm": f"DBSCAN(eps={eps:.2f})", "k": len(set(labels)) - (-1 in labels), **s})

res = P.table(pd.DataFrame(log), "autoresearch_log")
cand = res[(res["k"] >= 3) & (res["k"] <= 8) & (res["noise_share"] <= 0.05)]
best = cand.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
print(res.sort_values("silhouette", ascending=False).head(12).round(3).to_string(index=False))
print(f"\nSelected: {best['algorithm']} k={int(best['k'])} on {best['feature_set']} "
      f"(silhouette={best['silhouette']:.3f}, DB={best['davies_bouldin']:.3f})")
P.results["selected"] = {k: (v if isinstance(v, str) else float(v)) for k, v in best.dropna().items()}

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
km_rows = res[res["algorithm"] == "KMeans"]
for fs, g in km_rows.groupby("feature_set"):
    axes[0].plot(g["k"], g["inertia"], "o-", label=fs)
axes[0].set(title="K-Means elbow (inertia)", xlabel="k")
axes[0].legend()
for a, g in res[res["algorithm"].isin(["KMeans", "Agglomerative-Ward", "GaussianMixture"]) &
                (res["feature_set"] == best["feature_set"])].groupby("algorithm"):
    axes[1].plot(g["k"], g["silhouette"], "o-", label=a)
    axes[2].plot(g["k"], g["davies_bouldin"], "o-", label=a)
axes[1].set(title=f"Silhouette ({best['feature_set']}) ↑", xlabel="k")
axes[2].set(title=f"Davies–Bouldin ({best['feature_set']}) ↓", xlabel="k")
axes[1].legend()
axes[2].legend()
P.save(fig, "02_model_selection_curves")

# %% [markdown]
# ## 5 · Evaluation — stability, profiles, visualisation

# %%
P.section("5 · Evaluation")
X = scaled[best["feature_set"]]
k = int(best["k"])
algo = best["algorithm"]


def make(a, k, seed):
    if a == "KMeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed)
    if a == "GaussianMixture":
        return GaussianMixture(n_components=k, random_state=seed)
    return AgglomerativeClustering(n_clusters=k, linkage="ward")


if algo.startswith("DBSCAN"):  # unlikely given constraints, but keep the pipeline general
    algo = "KMeans"
final = make(algo, k, SEED)
labels = final.fit_predict(X)

# bootstrap stability: refit on 80 % resamples, compare on the overlap with ARI
rs = np.random.default_rng(SEED)
aris = []
for b in range(20):
    idx = rs.choice(len(X), size=int(0.8 * len(X)), replace=False)
    lb = make(algo, k, SEED + b + 1).fit_predict(X[idx])
    aris.append(adjusted_rand_score(labels[idx], lb))
P.results["stability_ari_mean"], P.results["stability_ari_std"] = float(np.mean(aris)), float(np.std(aris))
print(f"Bootstrap stability ARI = {np.mean(aris):.3f} ± {np.std(aris):.3f}")

cust["cluster"] = labels
prof = cust.groupby("cluster").agg(customers=("Recency", "size"), recency_days=("Recency", "median"),
                                   orders=("Frequency", "median"), revenue=("Monetary", "median"),
                                   total_revenue=("Monetary", "sum"), unique_products=("UniqueProducts", "median"))
prof["revenue_share"] = prof["total_revenue"] / prof["total_revenue"].sum()


# name segments by rank of recency (low = good), frequency and monetary
def name(r):
    rec, val = r["rec_rank"], r["val_rank"]
    if rec >= 0.66 and val >= 0.66:
        return "Champions"
    if rec >= 0.5 and val >= 0.33:
        return "Loyal / Potential"
    if rec < 0.5 and val >= 0.5:
        return "At-risk big spenders"
    if rec >= 0.5:
        return "New / Promising"
    return "Hibernating"


prof["rec_rank"] = (-prof["recency_days"]).rank(pct=True)
prof["val_rank"] = (prof["orders"].rank(pct=True) + prof["revenue"].rank(pct=True)) / 2
prof["segment"] = prof.apply(name, axis=1)
prof = prof.drop(columns=["rec_rank", "val_rank"])
print(prof.round(2))
P.table(prof.reset_index(), "segment_profiles")
P.results["segments"] = prof.round(3).reset_index().to_dict(orient="records")

pca = PCA(n_components=2, random_state=SEED).fit(X)
P2 = pca.transform(X)
T2 = TSNE(n_components=2, perplexity=40, random_state=SEED, init="pca").fit_transform(X)
seg_names = prof["segment"].to_dict()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for c in sorted(set(labels)):
    m = labels == c
    axes[0].scatter(P2[m, 0], P2[m, 1], s=6, alpha=.6, label=f"{c}: {seg_names[c]}")
    axes[1].scatter(T2[m, 0], T2[m, 1], s=6, alpha=.6, label=f"{c}: {seg_names[c]}")
axes[0].set_title(f"PCA projection ({pca.explained_variance_ratio_.sum():.0%} variance)")
axes[1].set_title("t-SNE projection")
axes[0].legend(markerscale=3, fontsize=8)
P.save(fig, "03_cluster_projections")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
snake = pd.DataFrame(X, columns=FEATURE_SETS[best["feature_set"]]).assign(cluster=labels).groupby("cluster").mean()
for c, row in snake.iterrows():
    axes[0].plot(row.index, row.values, "o-", label=f"{c}: {seg_names[c]}")
axes[0].axhline(0, c="k", lw=.8)
axes[0].set(title="Snake plot — standardized feature means per segment", ylabel="z-score")
axes[0].legend(fontsize=8)
share = pd.DataFrame({"customers": prof["customers"] / prof["customers"].sum(), "revenue": prof["revenue_share"]})
share.index = [f"{i}: {s}" for i, s in prof["segment"].items()]
share.plot.bar(ax=axes[1])
axes[1].set(title="Share of customers vs share of revenue", ylabel="share")
axes[1].tick_params(axis="x", rotation=20)
P.save(fig, "04_segment_profiles")

# %% [markdown]
# ## 6 · Deployment
# Segments are exported per customer (`outputs/customer_segments.csv`) for CRM upload; new customers can be assigned
# with `final.predict()` (K-Means / GMM) after the same log1p + scaling.

# %%
cust.reset_index()[["CustomerID", "Recency", "Frequency", "Monetary", "cluster"]].assign(
    segment=lambda d: d["cluster"].map(seg_names)).to_csv(P.out / "customer_segments.csv", index=False)

R = P.results
cust_share = prof["customers"] / prof["customers"].sum()
top_c, big_c = prof["revenue_share"].idxmax(), prof["customers"].idxmax()
paraphrase = (f"Segment {top_c} (**{prof.loc[top_c, 'segment']}**) is only {cust_share[top_c]:.0%} of customers but brings "
              f"{prof.loc[top_c, 'revenue_share']:.0%} of revenue, while the largest segment {big_c} "
              f"(**{prof.loc[big_c, 'segment']}**, {cust_share[big_c]:.0%} of customers) brings {prof.loc[big_c, 'revenue_share']:.0%}.")
P.finish(f"""# 03 · Customer Segmentation — Results

* {R['raw_rows']:,} transactions → **{R['n_customers']:,} customers** (RFM + behaviour features, log1p + standardised).
* AutoResearch evaluated {len(res)} configurations (K-Means, Agglomerative-Ward and GMM for k=2..10, plus a DBSCAN eps sweep, on 2 feature sets).
* Selected **{R['selected']['algorithm']} with k={int(R['selected']['k'])}** on *{R['selected']['feature_set']}*:
  silhouette **{R['selected']['silhouette']:.3f}**, Davies–Bouldin **{R['selected']['davies_bouldin']:.3f}**.
* Bootstrap stability (ARI over 20 resamples): **{R['stability_ari_mean']:.3f} ± {R['stability_ari_std']:.3f}**.

## Segment profiles
{prof.round(2).to_markdown()}

**Paraphrase:** the segments differ mainly in *how recently* and *how often* customers buy. {paraphrase}
Marketing should protect the high-value segment, re-activate lapsed big spenders, and nurture new customers toward a second purchase.
""")
