# %% [markdown]
# # 04 · Market-Basket Analysis — Associative Pattern Mining (Apriori vs FP-Growth)
# Replication of `dlmastery/data_science_examples/04_associative_pattern_mining`.
#
# **Original prompt:** *"Now lets do another project - associative pattern mining using popular kaggle data set - make sure
# you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and
# implement autoresearch to do hill climbing … include all details a data scientist and ai engineer will care."*
#
# **Data:** UCI / Kaggle *Online Retail* (UK gift retailer, 2010-2011). **Papers:** Agrawal & Srikant 1994 (Apriori),
# Han, Pei & Yin 2000 (FP-Growth), Brin et al. 1997 (lift / conviction).

# %%
import inspect
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_here.parent))
from _common import Project, load_online_retail, plt  # noqa: E402

import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

sns.set_theme(style="whitegrid")
P = Project(_here, "04 · Market Basket Analysis")

# %% [markdown]
# ## 1 · Business understanding
# Find product combinations bought together to drive **cross-sell recommendations, bundles and shelf/page placement**.
# Success = rules that are (a) strong (lift ≫ 1), (b) not trivially rare, and (c) **still hold on later, unseen
# transactions** — most tutorials skip (c).

# %%
P.section("2-3 · Data understanding & preparation")
raw = load_online_retail()
df = raw.dropna(subset=["Description"]).drop_duplicates()
df = df[~df["InvoiceNo"].astype(str).str.startswith("C") & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df = df[df["StockCode"].astype(str).str.match(r"^\d{5}")]  # drop POSTAGE, fees, manual adjustments, etc.
df = df[df["Country"] == "United Kingdom"].copy()
df["StockCode"] = df["StockCode"].astype(str)
names = df.groupby("StockCode")["Description"].agg(lambda s: s.str.strip().value_counts().index[0]).to_dict()
print(f"UK product lines: {len(df):,}, invoices: {df['InvoiceNo'].nunique():,}, products: {df['StockCode'].nunique():,}")

# temporal 3-way split: MINE Dec-2010..Jun-2011 | VALIDATE (tune thresholds) Jul..Sep | TEST (report only) Oct..Dec
cut_val, cut_test = pd.Timestamp("2011-07-01"), pd.Timestamp("2011-10-01")
inv_date = df.groupby("InvoiceNo")["InvoiceDate"].min()
mine_inv = inv_date[inv_date < cut_val].index
val_inv = inv_date[(inv_date >= cut_val) & (inv_date < cut_test)].index
hold_inv = inv_date[inv_date >= cut_test].index


def basket(frame, invoices, columns):
    """One-hot invoice × item boolean matrix restricted to `columns` (memory-friendly)."""
    f = frame[frame["InvoiceNo"].isin(invoices) & frame["StockCode"].isin(columns)][["InvoiceNo", "StockCode"]].drop_duplicates()
    b = pd.crosstab(f["InvoiceNo"], f["StockCode"]).astype(bool)
    b = b.reindex(index=pd.Index(invoices, name="InvoiceNo"), columns=list(columns), fill_value=False)
    return b.astype(bool)


mine_lines = df[df["InvoiceNo"].isin(mine_inv)]
item_support = (mine_lines.groupby("StockCode")["InvoiceNo"].nunique() / len(mine_inv)).sort_values(ascending=False)
keep = item_support[item_support >= 0.005].index  # prune ultra-rare items (makes Apriori tractable)
B_mine = basket(df, mine_inv, keep)
B_val = basket(df, val_inv, keep)
B_hold = basket(df, hold_inv, keep)
basket_size = B_mine.sum(axis=1)
P.results.update({"mining_invoices": int(len(B_mine)), "validation_invoices": int(len(B_val)),
                  "test_invoices": int(len(B_hold)), "items_kept": int(len(keep)),
                  "median_basket_size": float(basket_size.median())})
print(P.results)

fig, axes = plt.subplots(1, 3, figsize=(19, 5))
top = item_support.head(15)
axes[0].barh([names[c][:32] for c in top.index][::-1], top.values[::-1], color="#4c72b0")
axes[0].set(title="Top-15 items by support", xlabel="support")
axes[1].plot(np.arange(1, len(item_support) + 1), item_support.values)
axes[1].set(xscale="log", yscale="log", title="Long tail of item popularity", xlabel="item rank", ylabel="support")
axes[2].hist(basket_size.clip(upper=60), bins=60, color="#55a868")
axes[2].set(title="Distinct items per invoice (after pruning)", xlabel="items")
P.save(fig, "01_eda")


def make_rules(freq, n, metric="confidence", thr=0.3):
    if "num_itemsets" in inspect.signature(association_rules).parameters:  # mlxtend >= 0.23.2
        return association_rules(freq, num_itemsets=n, metric=metric, min_threshold=thr)
    return association_rules(freq, metric=metric, min_threshold=thr)


def holdout_metrics(rules, H, prefix="hold"):
    """Re-measure support / confidence / lift of each rule on later, unseen baskets."""
    Hv, col = H.values, {c: i for i, c in enumerate(H.columns)}
    out = []
    for a, c in zip(rules["antecedents"], rules["consequents"]):
        ai, ci = [col[x] for x in a], [col[x] for x in c]
        ma, mc = Hv[:, ai].all(axis=1), Hv[:, ci].all(axis=1)
        sa, sc, sac = ma.mean(), mc.mean(), (ma & mc).mean()
        conf = sac / sa if sa > 0 else np.nan
        out.append((sac, conf, conf / sc if sc > 0 and sa > 0 else np.nan))
    return pd.DataFrame(out, columns=[f"{prefix}_support", f"{prefix}_confidence", f"{prefix}_lift"], index=rules.index)


def survives(rules, prefix):
    return (rules[f"{prefix}_lift"] >= 2) & (rules[f"{prefix}_confidence"] >= 0.7 * rules["confidence"])


# %% [markdown]
# ## 4 · Modeling — AutoResearch over (min_support, min_confidence)
# Objective (measured on the **validation** quarter Jul–Sep) = number of **validated** rules (lift ≥ 2 and confidence
# ≥ 70 % of the mined confidence), tie-broken by mean validation lift. Oct–Dec stays untouched for the final report. Lower support finds more rules but also more spurious ones — the validation step
# measures that trade-off instead of guessing.

# %%
P.section("4 · AutoResearch")
log, cache = [], {}
for ms in (0.03, 0.02, 0.015, 0.01):
    t = time.time()
    freq = fpgrowth(B_mine, min_support=ms, use_colnames=True)
    t_fp = time.time() - t
    cache[ms] = freq
    for mc in (0.3, 0.5, 0.7):
        rules = make_rules(freq, len(B_mine), "confidence", mc)
        if rules.empty:
            log.append({"min_support": ms, "min_confidence": mc, "itemsets": len(freq), "rules": 0, "validated": 0})
            continue
        h = rules.join(holdout_metrics(rules, B_val, "val"))
        valid = survives(h, "val")
        log.append({"min_support": ms, "min_confidence": mc, "itemsets": len(freq), "rules": len(rules),
                    "validated": int(valid.sum()), "validated_share": float(valid.mean()),
                    "mean_val_lift": float(h["val_lift"].mean()), "fpgrowth_seconds": t_fp})
        print(f"support={ms:<6} conf={mc:<4} itemsets={len(freq):5d} rules={len(rules):5d} "
              f"validated={int(valid.sum()):5d} ({valid.mean():.0%})")
res = P.table(pd.DataFrame(log), "autoresearch_log")
best = res.sort_values(["validated", "mean_val_lift"], ascending=False).iloc[0]
MS, MC = float(best["min_support"]), float(best["min_confidence"])
P.results["selected"] = {"min_support": MS, "min_confidence": MC, "rules": int(best["rules"]),
                         "validated": int(best["validated"])}
print(f"\nSelected min_support={MS}, min_confidence={MC}")

# Apriori vs FP-Growth: identical output, different speed
t = time.time()
f_ap = apriori(B_mine, min_support=MS, use_colnames=True, low_memory=True)
t_ap = time.time() - t
t = time.time()
f_fp = fpgrowth(B_mine, min_support=MS, use_colnames=True)
t_fp = time.time() - t
same = set(map(frozenset, f_ap["itemsets"])) == set(map(frozenset, f_fp["itemsets"]))
P.results["apriori_vs_fpgrowth"] = {"apriori_s": round(t_ap, 2), "fpgrowth_s": round(t_fp, 2),
                                    "n_itemsets": int(len(f_fp)), "identical": bool(same)}
print(P.results["apriori_vs_fpgrowth"])

# %% [markdown]
# ## 5 · Evaluation of the final rule set

# %%
P.section("5 · Evaluation")
rules = make_rules(f_fp, len(B_mine), "confidence", MC)
rules = rules.join(holdout_metrics(rules, B_val, "val")).join(holdout_metrics(rules, B_hold, "hold"))
rules["validated"] = survives(rules, "val")  # decided on validation data only
rules["survives_test"] = survives(rules, "hold")  # reported, never used for decisions
P.results["validated_rules"] = int(rules["validated"].sum())
P.results["validated_rules_surviving_test"] = float(rules.loc[rules["validated"], "survives_test"].mean()) if rules["validated"].any() else float("nan")
lab = lambda s: " + ".join(names[x][:28] for x in sorted(s))  # noqa: E731
rules["rule"] = rules["antecedents"].map(lab) + "  →  " + rules["consequents"].map(lab)
show = rules.sort_values("lift", ascending=False)[["rule", "support", "confidence", "lift", "conviction",
                                                   "val_lift", "validated", "hold_confidence", "hold_lift",
                                                   "survives_test"]]
print(show.head(15).round(3).to_string(index=False))
P.table(show.round(4), "rules")
P.results["top_rules"] = show.head(10).round(3).to_dict(orient="records")
P.results["holdout_lift_correlation"] = float(rules[["lift", "hold_lift"]].corr(method="spearman").iloc[0, 1])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc = axes[0].scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", s=18, alpha=.8)
fig.colorbar(sc, ax=axes[0], label="lift")
axes[0].set(xlabel="support", ylabel="confidence", title=f"{len(rules)} rules (min_sup={MS}, min_conf={MC})")
axes[1].scatter(rules["lift"], rules["hold_lift"], c=np.where(rules["validated"], "#55a868", "#c44e52"), s=14, alpha=.7)
mx = np.nanmax(rules[["lift", "hold_lift"]].values)
axes[1].plot([0, mx], [0, mx], "k--", lw=1)
axes[1].set(xlabel="lift (mined Dec–Jun)", ylabel="lift (test quarter Oct–Dec)",
            title=f"Do rules generalise? Spearman ρ={P.results['holdout_lift_correlation']:.2f} (green = validated on Jul–Sep)")
P.save(fig, "02_rules_quality")

# rule network of the top-25 validated rules (manual circular layout, no networkx needed)
net = rules[rules["validated"]].sort_values("lift", ascending=False).head(25)
nodes = sorted({x for s in pd.concat([net["antecedents"], net["consequents"]]) for x in s})
if nodes:
    ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, ang)}
    fig, ax = plt.subplots(figsize=(13, 13))
    for _, r in net.iterrows():
        for a in r["antecedents"]:
            for c in r["consequents"]:
                ax.annotate("", xy=pos[c], xytext=pos[a], arrowprops=dict(arrowstyle="->", alpha=.5,
                            lw=0.5 + r["lift"] / net["lift"].max() * 2.5, color="#4c72b0"))
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=120, c="#dd8452", zorder=3)
        ax.text(x * 1.08, y * 1.08, names[n][:26], fontsize=8, ha="left" if x >= 0 else "right", va="center")
    ax.set(xlim=(-1.8, 1.8), ylim=(-1.4, 1.4), title="Top validated association rules (edge width ∝ lift)")
    ax.axis("off")
    P.save(fig, "03_rule_network")

# %% [markdown]
# ## 6 · Deployment — "customers who bought this also bought" recommender

# %%
P.section("6 · Recommender demo")
val_rules = rules[rules["validated"]]


def recommend(cart: set, k: int = 5):
    hits = val_rules[val_rules["antecedents"].map(lambda a: a.issubset(cart))]
    hits = hits[~hits["consequents"].map(lambda c: c.issubset(cart))]
    hits = hits.assign(consequents=hits["consequents"].map(list))
    recs = (hits.explode("consequents").groupby("consequents")["confidence"].max()
            .sort_values(ascending=False).head(k))
    return [(names[i], round(c, 3)) for i, c in recs.items()]


popular_antecedents = val_rules.sort_values("support", ascending=False)["antecedents"].head(3).tolist()
demo = []
for cart in popular_antecedents:
    demo.append({"cart": " + ".join(names[x] for x in cart), "recommendations": recommend(set(cart))})
    print(demo[-1])
P.results["recommender_demo"] = demo

R = P.results
top_rule = show.iloc[0]
val_share = rules["validated"].mean()
P.finish(f"""# 04 · Market Basket Analysis — Results

* UK invoices: {R['mining_invoices']:,} for mining (Dec-2010 → Jun-2011), {R['validation_invoices']:,} for **validation / threshold tuning**
  (Jul → Sep-2011), {R['test_invoices']:,} for the **untouched test quarter** (Oct → Dec-2011).
  {R['items_kept']:,} items with support ≥ 0.5 %; median basket = {R['median_basket_size']:.0f} distinct items.
* AutoResearch grid (scored on validation): selected **support={R['selected']['min_support']}, confidence={R['selected']['min_confidence']}**,
  {R['selected']['rules']} rules, **{R['selected']['validated']} validated** (lift ≥ 2, confidence retained ≥ 70 %).
* Of the validated rules, **{R['validated_rules_surviving_test']:.0%} still pass the same test in the unseen Oct–Dec quarter**.
* Apriori vs FP-Growth: identical itemsets = **{R['apriori_vs_fpgrowth']['identical']}**; time {R['apriori_vs_fpgrowth']['apriori_s']}s vs {R['apriori_vs_fpgrowth']['fpgrowth_s']}s.
* Rank agreement of lift between mining and test periods: Spearman ρ = {R['holdout_lift_correlation']:.2f}.

## Top rules by lift
{show.head(12).round(3).to_markdown(index=False)}

**Paraphrase:** the strongest rule is *{top_rule['rule']}*: buying the left side makes the right side
{top_rule['lift']:.0f}× more likely than chance (confidence {top_rule['confidence']:.0%} when mined, {top_rule['hold_confidence']:.0%} in the test quarter).
{val_share:.0%} of the final rules passed validation, and {R['validated_rules_surviving_test']:.0%} of those also held up in the
holiday quarter that was never used for any decision. Those rules are reasonable candidates for "frequently bought together"
widgets and bundles. Rules that failed validation are treated as coincidences and are not used by the recommender.
""")
