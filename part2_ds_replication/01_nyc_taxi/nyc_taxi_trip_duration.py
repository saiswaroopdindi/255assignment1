# %% [markdown]
# # 01 · NYC Taxi Trip-Duration Prediction (CRISP-DM + AutoResearch hill-climbing)
# Replication of `dlmastery/data_science_examples/01_nyc_taxi_trip_prediction`.
#
# **Original prompt:** *"You will do end2end a data science project including data, training, deployment, crisp-dm
# framework… you can use Kaggle NYC taxi challenge… make sure frontend includes … trip estimation."* +
# *"follow the autoresearch repo skills diligently for hill climbing."*
#
# **Data:** NYC TLC Yellow Taxi trip records, January 2024 (public, same source as the Kaggle challenge, ~3 M trips) +
# taxi-zone lookup. The Kaggle competition file needs a login, so the official TLC open-data parquet is used instead.
#
# **Target:** trip duration in minutes, modelled as `log1p(minutes)` → metric **RMSLE** (the Kaggle metric).

# %%
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_here.parent))
from _common import Project, download, plt  # noqa: E402

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid")
P = Project(_here, "01 · NYC Taxi Trip Duration")

# %% [markdown]
# ## 1 · Business understanding
# Riders and dispatchers need an ETA at booking time. Only information known **at pickup** may be used (no drop-off time,
# no fare/tip, which are only known after the trip). Success = beat a distance-only baseline on RMSLE on a later, unseen week.

# %%
P.section("1-2 · Business & data understanding")
trips_path = download("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet")
zones_path = download("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv")
cols = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance",
        "RatecodeID", "PULocationID", "DOLocationID"]
raw = pd.read_parquet(trips_path, columns=cols)
zones = pd.read_csv(zones_path)
print(raw.shape)
print(raw.describe().T.round(2))
P.results["raw_rows"] = int(len(raw))

# %% [markdown]
# ## 2 · Data understanding & quality
# Typical TLC data issues: negative/zero durations, trips > 3 h (meter left on), zero distance, timestamps outside the month.

# %%
df = raw.copy()
df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
checks = {
    "duration <= 1 min": (df["duration_min"] <= 1).mean(),
    "duration > 180 min": (df["duration_min"] > 180).mean(),
    "distance <= 0.1 mi": (df["trip_distance"] <= 0.1).mean(),
    "distance > 100 mi": (df["trip_distance"] > 100).mean(),
    "pickup outside Jan-2024": (~df["tpep_pickup_datetime"].between("2024-01-01", "2024-02-01")).mean(),
    "passenger_count missing": df["passenger_count"].isna().mean(),
    "implied speed > 80 mph": ((df["trip_distance"] / (df["duration_min"] / 60)) > 80).mean(),
}
quality = pd.DataFrame({"check": checks.keys(), "share_of_rows": checks.values()})
print(quality.round(4).to_string(index=False))
P.table(quality, "data_quality")

mask = (df["duration_min"].between(1, 180) & df["trip_distance"].between(0.1, 100)
        & df["tpep_pickup_datetime"].between("2024-01-01", "2024-02-01")
        & ((df["trip_distance"] / (df["duration_min"] / 60)) <= 80))
df = df.loc[mask].copy()
P.results["clean_rows"] = int(len(df))
P.results["removed_share"] = float(1 - len(df) / len(raw))
print(f"Kept {len(df):,} rows ({P.results['removed_share']:.1%} removed)")

# work on a reproducible 600k-trip sample to respect limited compute
df = df.sample(n=min(600_000, len(df)), random_state=SEED).sort_values("tpep_pickup_datetime").reset_index(drop=True)

# %% [markdown]
# ## 3 · Data preparation & feature engineering
# Feature groups (each is a *hypothesis* tested by the hill-climber):
# * **G0 distance** — `log_distance`
# * **G1 time** — hour, day-of-week, weekend, cyclic hour encoding
# * **G2 zone ids** — pickup / drop-off TLC zone ids
# * **G3 geography** — boroughs, airport flags, same-zone, rate code
# * **G4 target encoding** — out-of-fold mean `log_duration` per pickup zone, drop-off zone and zone *pair* (fit on train only)
# * **G5 misc** — vendor, passenger count

# %%
P.section("3 · Data preparation")
zmap_b = zones.assign(Borough=zones["Borough"].fillna("Unknown")).set_index("LocationID")["Borough"].to_dict()
BOROUGHS = sorted(set(zmap_b.values()) | {"Unknown"})  # fixed category order shared by training and serving
AIRPORTS = {1: "EWR", 132: "JFK", 138: "LGA"}
df["y"] = np.log1p(df["duration_min"])
df["log_distance"] = np.log1p(df["trip_distance"])
ts = df["tpep_pickup_datetime"]
df["hour"], df["dow"] = ts.dt.hour, ts.dt.dayofweek
df["is_weekend"] = (df["dow"] >= 5).astype(int)
df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * (ts.dt.hour + ts.dt.minute / 60) / 24), np.cos(2 * np.pi * (ts.dt.hour + ts.dt.minute / 60) / 24)
for side in ("PU", "DO"):
    b = df[f"{side}LocationID"].map(zmap_b).fillna("Unknown")
    df[f"{side}_borough"] = pd.Categorical(b, categories=BOROUGHS).codes
    df[f"{side}_manhattan"] = (b == "Manhattan").astype(int)
    df[f"{side}_airport"] = df[f"{side}LocationID"].isin(AIRPORTS).astype(int)
df["same_zone"] = (df["PULocationID"] == df["DOLocationID"]).astype(int)
df["RatecodeID"] = df["RatecodeID"].fillna(1).clip(upper=6)
df["passenger_count"] = df["passenger_count"].fillna(1)
df["pair"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

GROUPS = {
    "G0 distance": ["log_distance"],
    "G1 time": ["hour", "dow", "is_weekend", "hour_sin", "hour_cos"],
    "G2 zone ids": ["PULocationID", "DOLocationID"],
    "G3 geography": ["PU_borough", "DO_borough", "PU_manhattan", "DO_manhattan", "PU_airport", "DO_airport",
                     "same_zone", "RatecodeID"],
    "G4 target enc": ["te_pu", "te_do", "te_pair"],
    "G5 misc": ["VendorID", "passenger_count"],
}
TE_KEYS = {"te_pu": "PULocationID", "te_do": "DOLocationID", "te_pair": "pair"}


def target_encode(train: pd.DataFrame, others: list, k: float = 20.0):
    """Smoothed mean encoding. Train rows get 5-fold out-of-fold values (no self-leakage)."""
    train, others = train.copy(), [o.copy() for o in others]
    gm = train["y"].mean()
    for col, key in TE_KEYS.items():
        oof = np.full(len(train), gm)
        for tr_idx, va_idx in KFold(5, shuffle=True, random_state=SEED).split(train):
            g = train.iloc[tr_idx].groupby(key)["y"].agg(["sum", "count"])
            enc = (g["sum"] + k * gm) / (g["count"] + k)
            oof[va_idx] = train.iloc[va_idx][key].map(enc).fillna(gm).values
        train[col] = oof
        g = train.groupby(key)["y"].agg(["sum", "count"])
        enc = (g["sum"] + k * gm) / (g["count"] + k)
        for o in others:
            o[col] = o[key].map(enc).fillna(gm)
    return train, others


# temporal split: last 8 days = TEST (never touched until the end); previous 7 days = VALIDATION for hill-climbing
test_start, val_start = pd.Timestamp("2024-01-24"), pd.Timestamp("2024-01-17")
train_full = df[df["tpep_pickup_datetime"] < test_start]
test = df[df["tpep_pickup_datetime"] >= test_start]
sub_train = train_full[train_full["tpep_pickup_datetime"] < val_start]
val = train_full[train_full["tpep_pickup_datetime"] >= val_start]
sub_train, (val,) = target_encode(sub_train, [val])
print(f"sub-train {len(sub_train):,} | val {len(val):,} | test {len(test):,}")
P.results["split"] = {"sub_train": len(sub_train), "val": len(val), "train_full": len(train_full), "test": len(test)}

# EDA figures
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
axes[0].hist(df["duration_min"], bins=100, color="#4c72b0")
axes[0].set(yscale="log", xlabel="minutes", title="Trip duration (log count)")
hb = axes[1].hexbin(df["trip_distance"].clip(upper=25), df["duration_min"].clip(upper=90), gridsize=60, bins="log", cmap="viridis")
axes[1].set(xlabel="distance (mi, clipped 25)", ylabel="minutes (clipped 90)", title="Distance vs duration")
fig.colorbar(hb, ax=axes[1])
heat = df.pivot_table(index="dow", columns="hour", values="duration_min", aggfunc="median")
sns.heatmap(heat, cmap="magma", ax=axes[2], cbar_kws={"label": "median minutes"})
axes[2].set(title="Median duration: day-of-week × hour", yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
P.save(fig, "01_eda_overview")

fig, ax = plt.subplots(figsize=(7, 5.5))
bb = df.assign(PU=df["PULocationID"].map(zmap_b), DO=df["DOLocationID"].map(zmap_b))
piv = bb.pivot_table(index="PU", columns="DO", values="duration_min", aggfunc="median")
sns.heatmap(piv, annot=True, fmt=".0f", cmap="rocket_r", ax=ax)
ax.set_title("Median minutes by borough pair")
P.save(fig, "02_borough_pair_durations")

# %% [markdown]
# ## 4 · Modeling — AutoResearch hill-climbing
# Loop: *propose* (add a feature group / change a hyper-parameter) → *train on sub-train* → *score on validation* →
# **keep only if RMSLE improves**. Every experiment is logged, including the rejected ones.

# %%
P.section("4 · AutoResearch hill-climbing")


def rmsle(y_true_log, y_pred_log):
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))


def fit_eval(feats, params, tr, va):
    m = HistGradientBoostingRegressor(random_state=SEED, **params).fit(tr[feats], tr["y"])
    return rmsle(va["y"], m.predict(va[feats])), m


log = []
base_params = {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 50}
current_feats = GROUPS["G0 distance"]
best_score, _ = fit_eval(current_feats, base_params, sub_train, val)
log.append({"step": 0, "proposal": "baseline: distance only", "val_rmsle": best_score, "kept": True})
print(f"step 0  distance-only  RMSLE={best_score:.4f}")
step = 1
for gname in ["G1 time", "G2 zone ids", "G3 geography", "G4 target enc", "G5 misc"]:
    cand = current_feats + GROUPS[gname]
    s, _ = fit_eval(cand, base_params, sub_train, val)
    kept = s < best_score - 1e-4
    log.append({"step": step, "proposal": f"+ {gname}", "val_rmsle": s, "kept": kept})
    print(f"step {step}  + {gname:14s} RMSLE={s:.4f}  {'KEEP' if kept else 'reject'}")
    if kept:
        current_feats, best_score = cand, s
    step += 1

# hyper-parameter hill-climbing (random proposals around the incumbent)
best_params = dict(base_params)
for i in range(10):
    prop = dict(best_params)
    prop["learning_rate"] = float(np.clip(best_params["learning_rate"] * rng.choice([0.5, 0.75, 1.5]), 0.02, 0.3))
    prop["max_leaf_nodes"] = int(np.clip(best_params["max_leaf_nodes"] * rng.choice([0.5, 1, 2]), 15, 255))
    prop["min_samples_leaf"] = int(rng.choice([20, 50, 100, 200]))
    prop["max_iter"] = int(np.clip(best_params["max_iter"] * rng.choice([1, 1.5, 2]), 200, 1200))
    s, _ = fit_eval(current_feats, prop, sub_train, val)
    kept = s < best_score - 1e-4
    log.append({"step": step, "proposal": f"HPO {prop}", "val_rmsle": s, "kept": kept})
    print(f"step {step}  HPO lr={prop['learning_rate']:.3f} leaves={prop['max_leaf_nodes']} "
          f"msl={prop['min_samples_leaf']} iters={prop['max_iter']}  RMSLE={s:.4f}  {'KEEP' if kept else 'reject'}")
    if kept:
        best_params, best_score = prop, s
    step += 1
hill = P.table(pd.DataFrame(log), "autoresearch_log")
P.results["hill_climb"] = {"final_features": current_feats, "best_params": best_params, "best_val_rmsle": best_score,
                           "n_experiments": len(log)}

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(hill["step"], hill["val_rmsle"], "o", color="lightgray", label="all proposals")
ax.plot(hill["step"], hill["val_rmsle"].where(hill["kept"]).cummin().ffill(), "-", c="#c44e52", lw=2, label="incumbent (kept)")
ax.set(xlabel="experiment", ylabel="validation RMSLE", title="AutoResearch hill-climb trajectory")
ax.legend()
P.save(fig, "03_hill_climb")

# %% [markdown]
# ## 5 · Evaluation on the untouched final week
# Re-fit target encodings on the full training period, train all candidate models, score the test week once.

# %%
P.section("5 · Evaluation")
train_full, (test,) = target_encode(train_full, [test])
feats = current_feats
y_tr, y_te = train_full["y"].values, test["y"].values
preds = {}
preds["Baseline: global mean"] = np.full(len(test), y_tr.mean())
bins = pd.qcut(train_full["trip_distance"], 50, duplicates="drop", retbins=True)[1]
bins[0], bins[-1] = -np.inf, np.inf
med = train_full.groupby(pd.cut(train_full["trip_distance"], bins), observed=False)["y"].median()
preds["Baseline: distance-bucket median"] = pd.cut(test["trip_distance"], bins).map(med).astype(float).fillna(y_tr.mean()).values
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(train_full[feats], y_tr)
preds["Ridge regression"] = ridge.predict(test[feats])
rf_idx = rng.choice(len(train_full), size=min(150_000, len(train_full)), replace=False)
rf = RandomForestRegressor(n_estimators=120, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
rf.fit(train_full.iloc[rf_idx][feats], y_tr[rf_idx])
preds["Random forest (150k sample)"] = rf.predict(test[feats])
hgb = HistGradientBoostingRegressor(random_state=SEED, **best_params).fit(train_full[feats], y_tr)
preds["HistGradientBoosting (hill-climbed)"] = hgb.predict(test[feats])
try:
    from xgboost import XGBRegressor

    xgb = XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                       n_jobs=-1, random_state=SEED).fit(train_full[feats], y_tr)
    preds["XGBoost"] = xgb.predict(test[feats])
except Exception as e:  # not installed, or libomp missing on macOS (brew install libomp)
    print(f"xgboost unavailable ({type(e).__name__}) - skipped")

rows = []
for name, p in preds.items():
    mins_true, mins_pred = np.expm1(y_te), np.expm1(p)
    rows.append({"model": name, "test_RMSLE": rmsle(y_te, p), "MAE_minutes": float(np.mean(np.abs(mins_true - mins_pred))),
                 "MedAE_minutes": float(np.median(np.abs(mins_true - mins_pred))),
                 "R2_log": float(1 - np.sum((y_te - p) ** 2) / np.sum((y_te - y_te.mean()) ** 2))})
board = P.table(pd.DataFrame(rows).sort_values("test_RMSLE"), "test_leaderboard")
print(board.round(4).to_string(index=False))
P.results["test_leaderboard"] = board.round(4).to_dict(orient="records")
best_name = board.iloc[0]["model"]
best_pred = preds[best_name]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
board.set_index("model")["test_RMSLE"].sort_values().plot.barh(ax=axes[0], color="#55a868")
axes[0].set(title="Test RMSLE (lower is better)", xlabel="RMSLE")
axes[1].hexbin(np.expm1(y_te).clip(max=90), np.expm1(best_pred).clip(max=90), gridsize=60, bins="log", cmap="viridis")
axes[1].plot([0, 90], [0, 90], "r--")
axes[1].set(xlabel="actual minutes", ylabel="predicted minutes", title=f"Predicted vs actual — {best_name}")
res = pd.DataFrame({"hour": test["hour"].values, "abs_err": np.abs(np.expm1(y_te) - np.expm1(best_pred))})
res.groupby("hour")["abs_err"].mean().plot(ax=axes[2], marker="o")
axes[2].set(xlabel="pickup hour", ylabel="MAE (min)", title="Error by hour of day")
P.save(fig, "04_evaluation")

samp = test.sample(20_000, random_state=SEED)
pi = permutation_importance(hgb, samp[feats], samp["y"], n_repeats=5, random_state=SEED, n_jobs=-1,
                            scoring="neg_root_mean_squared_error")
imp = pd.Series(pi.importances_mean, index=feats).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
imp.plot.barh(ax=ax, color="#8172b3")
ax.set(title="Permutation importance (test sample, HGB)", xlabel="increase in RMSE (log space)")
P.save(fig, "05_permutation_importance")
P.results["top_features"] = imp.sort_values(ascending=False).head(6).round(4).to_dict()

# %% [markdown]
# ## 6 · Deployment — trip estimator
# The same feature builder + model can sit behind a REST endpoint; here is the core `estimate()` function.

# %%
P.section("6 · Deployment demo")
zone_name = zones.set_index("LocationID")["Zone"].to_dict()
gm = y_tr.mean()
enc_tables = {}
for col, key in TE_KEYS.items():  # identical smoothing (k=20) to the encodings the model was trained on
    g = train_full.groupby(key)["y"].agg(["sum", "count"])
    enc_tables[col] = (g["sum"] + 20.0 * gm) / (g["count"] + 20.0)


def estimate(pu: int, do: int, miles: float, when: str) -> float:
    t = pd.Timestamp(when)
    r = {"log_distance": np.log1p(miles), "hour": t.hour, "dow": t.dayofweek, "is_weekend": int(t.dayofweek >= 5),
         "hour_sin": np.sin(2 * np.pi * (t.hour + t.minute / 60) / 24), "hour_cos": np.cos(2 * np.pi * (t.hour + t.minute / 60) / 24),
         "PULocationID": pu, "DOLocationID": do, "same_zone": int(pu == do), "RatecodeID": 2 if 132 in (pu, do) else 1,
         "VendorID": 2, "passenger_count": 1}
    for side, z in (("PU", pu), ("DO", do)):
        b = zmap_b.get(z, "Unknown")
        r[f"{side}_borough"] = BOROUGHS.index(b)
        r[f"{side}_manhattan"], r[f"{side}_airport"] = int(b == "Manhattan"), int(z in AIRPORTS)
    r["te_pu"] = enc_tables["te_pu"].get(pu, gm)
    r["te_do"] = enc_tables["te_do"].get(do, gm)
    r["te_pair"] = enc_tables["te_pair"].get(f"{pu}_{do}", gm)
    return float(np.expm1(hgb.predict(pd.DataFrame([r])[feats])[0]))


examples = [(132, 161, 17.0, "2024-01-25 17:30"), (161, 236, 2.1, "2024-01-25 08:45"),
            (138, 230, 9.5, "2024-01-27 23:10"), (79, 107, 0.9, "2024-01-26 13:00")]
demo = pd.DataFrame([{"from": zone_name.get(a), "to": zone_name.get(b), "miles": m, "when": w,
                      "eta_minutes": round(estimate(a, b, m, w), 1)} for a, b, m, w in examples])
print(demo.to_string(index=False))
P.table(demo, "eta_examples")
P.results["eta_examples"] = demo.to_dict(orient="records")

# %%
R = P.results
P.finish(f"""# 01 · NYC Taxi Trip Duration — Results

* Data: {R['raw_rows']:,} TLC trips (Jan 2024) → {R['clean_rows']:,} after cleaning ({R['removed_share']:.1%} removed); 600k-trip sample modelled.
* Split: **temporal** — train Jan 1–23, test Jan 24–31 (validation week Jan 17–23 for hill-climbing). No random shuffling → no look-ahead.
* AutoResearch: {R['hill_climb']['n_experiments']} logged experiments; final validation RMSLE **{R['hill_climb']['best_val_rmsle']:.4f}**.
  Kept feature set: `{', '.join(R['hill_climb']['final_features'])}`.

## Test-week leaderboard
{board.round(4).to_markdown(index=False)}

## Top features (permutation importance)
{chr(10).join(f'- `{k}`: {v}' for k, v in R['top_features'].items())}

## ETA demo
{demo.to_markdown(index=False)}

**Paraphrase:** distance explains most of the trip time, but *when* (rush hour, weekday) and *where* (zone pair, airports)
add a large improvement over a distance-only rule. Target-encoding the zone pair learns typical route congestion and was
built out-of-fold to avoid leakage.
""")
