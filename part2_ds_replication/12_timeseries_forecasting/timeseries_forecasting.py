# %% [markdown]
# # 12 · Time-Series Forecasting — Daily Bike-Share Demand (CRISP-DM, rolling-origin backtest)
# Replication of `dlmastery/data_science_examples/12_timeseries_forecasting` (TimePulse).
#
# **Original prompt:** *"build a similar website … for timeseries forecasting similar to other projects you built -
# including detailed crisp-dm steps and admin and other dashboards"* (the reference app shows multi-horizon forecast
# fans and 40-lag ACF/PACF).
#
# **Data:** UCI / Kaggle *Bike Sharing Dataset* (Capital Bikeshare, Washington D.C., 2011-2012, 731 days; Fanaee-T &
# Gama 2013). Target = daily rentals `cnt`. Horizon = **14 days**.

# %%
import sys
import warnings
import zipfile
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_here.parent))
from _common import DATA, Project, download, plt  # noqa: E402

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
SEED = 42
H = 14
sns.set_theme(style="whitegrid")
P = Project(_here, "12 · Time-Series Forecasting (bike-share demand)")

# %% [markdown]
# ## 1 · Business understanding
# The operator rebalances bikes and schedules staff two weeks ahead. Success = lower error than a **seasonal-naive**
# forecast (same weekday last week) across many forecast origins — measured with **MASE** (< 1 means better than naive),
# plus MAE / RMSE / sMAPE and empirical (backtest-based) prediction intervals.

# %%
P.section("2 · Data understanding")
z = download("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", "bike_sharing.zip")
with zipfile.ZipFile(z) as zf:
    zf.extract("day.csv", DATA)
day = pd.read_csv(DATA / "day.csv", parse_dates=["dteday"]).set_index("dteday").asfreq("D")
y = day["cnt"].astype(float)
print(day.shape, y.isna().sum(), "missing days")
print(y.describe().round(1))
P.results["n_days"] = int(len(y))

adf_p = adfuller(y)[1]
kpss_p = kpss(y, regression="c", nlags="auto")[1]
adf_p_diff = adfuller(y.diff().dropna())[1]
P.results["stationarity"] = {"adf_p_level": float(adf_p), "kpss_p_level": float(kpss_p), "adf_p_first_diff": float(adf_p_diff)}
print(P.results["stationarity"])

stl = STL(y, period=7, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
for ax, (lbl, s) in zip(axes, [("observed", y), ("trend", stl.trend), ("weekly seasonal", stl.seasonal), ("residual", stl.resid)]):
    ax.plot(s, lw=1)
    ax.set_ylabel(lbl)
axes[0].set_title("STL decomposition (period = 7 days)")
P.save(fig, "01_stl_decomposition")

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(y.diff().dropna(), lags=40, ax=axes[0], title="ACF of Δcnt (40 lags)")
plot_pacf(y.diff().dropna(), lags=40, ax=axes[1], title="PACF of Δcnt (40 lags)", method="ywm")
P.save(fig, "02_acf_pacf")

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
sns.boxplot(x=np.asarray(day.index.dayofweek), y=y.values, ax=axes[0], color="#4c72b0")
axes[0].set(title="Rentals by weekday (0=Mon)", xlabel="")
sns.boxplot(x=np.asarray(day.index.month), y=y.values, ax=axes[1], color="#55a868")
axes[1].set(title="Rentals by month", xlabel="")
sns.scatterplot(x=day["temp"] * 41, y=y, hue=day["weathersit"], palette="viridis", ax=axes[2], s=18)
axes[2].set(title="Temperature (°C) vs rentals, colour = weather", xlabel="°C")
P.save(fig, "03_seasonality_drivers")

# %% [markdown]
# **Findings:** strong upward trend (system growth 2011→2012), annual seasonality (summer peak), weaker weekly pattern,
# big negative shocks on bad-weather days. ADF says the level is non-stationary, while the first difference is stationary.
#
# ## 3 · Data preparation
# Calendar features that are **known in advance** (day-of-week, month, holiday, working day, Fourier terms of
# day-of-year, time index) + lagged target values ≥ 1 day. Weather is *not* used because it would not be known 14 days
# ahead (that would be leakage).

# %%
P.section("3-4 · Feature engineering & models")
from pandas.tseries.holiday import USFederalHolidayCalendar


def calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features — computable for any future date."""
    hol = set(USFederalHolidayCalendar().holidays(idx.min(), idx.max()))
    c = pd.DataFrame(index=idx)
    c["dow"], c["month"] = idx.dayofweek, idx.month
    c["holiday"] = [int(d in hol) for d in idx]
    c["workingday"] = ((c["dow"] < 5) & (c["holiday"] == 0)).astype(int)
    doy = idx.dayofyear
    for k in (1, 2):
        c[f"sin{k}"], c[f"cos{k}"] = np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)
    c["t"] = (idx - y.index[0]).days
    return c


cal = calendar(pd.date_range(y.index[0], y.index[-1] + pd.Timedelta(days=60), freq="D"))
LAGS = [1, 2, 3, 7, 14]


def gbm_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Recursive multi-step forecast with lag + calendar features (gradient boosting)."""
    frame = cal.loc[train.index].copy()
    for L in LAGS:
        frame[f"lag{L}"] = train.shift(L)
    frame["roll7"] = train.shift(1).rolling(7).mean()
    frame = frame.dropna()
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED)
    # trend: tree models can't extrapolate -> learn the residual from a 28-day rolling level
    level = train.rolling(28, min_periods=7).mean().shift(1).loc[frame.index]
    m.fit(frame.assign(level=level), train.loc[frame.index] - level)
    hist = train.copy()
    out = []
    for d in pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h, freq="D"):
        r = cal.loc[[d]].copy()
        for L in LAGS:
            r[f"lag{L}"] = hist.iloc[-L]
        r["roll7"] = hist.iloc[-7:].mean()
        lvl = hist.iloc[-28:].mean()
        pred = float(m.predict(r.assign(level=lvl))[0] + lvl)
        out.append(pred)
        hist.loc[d] = pred
    return np.array(out)


MODELS = {
    "Naive (last value)": lambda tr, h: np.repeat(tr.iloc[-1], h),
    "Seasonal naive (t-7)": lambda tr, h: np.resize(tr.iloc[-7:].values, h),
    "Moving average (28d)": lambda tr, h: np.repeat(tr.iloc[-28:].mean(), h),
    "Holt-Winters (add. trend, damped, weekly)": lambda tr, h: ExponentialSmoothing(
        tr, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7).fit().forecast(h).values,
    "SARIMA(1,1,1)(1,0,1,7)": lambda tr, h: SARIMAX(tr, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(
        disp=False).forecast(h).values,
    "Gradient boosting (lags + calendar)": gbm_forecast,
}

# %% [markdown]
# ## 4-5 · Rolling-origin backtest (expanding window)
# 12 forecast origins spaced 14 days apart over the last ~6 months; each model forecasts 14 days ahead from each origin.
# Metrics are averaged across origins. MASE is scaled by the in-sample seasonal-naive MAE.

# %%
origins = [len(y) - H - 14 * i for i in range(12)][::-1]
recs, fan_store = [], {}
for o in origins:
    tr, te = y.iloc[:o], y.iloc[o:o + H]
    scale = np.mean(np.abs(tr.values[7:] - tr.values[:-7]))
    for name, f in MODELS.items():
        try:
            p = np.asarray(f(tr, H), dtype=float)
        except Exception as e:  # keep the backtest robust to an occasional convergence failure
            print(f"  {name} failed at origin {o}: {e}")
            continue
        err = te.values - p
        recs.append({"origin": te.index[0].date(), "model": name, "MAE": np.mean(np.abs(err)),
                     "RMSE": np.sqrt(np.mean(err ** 2)),
                     "sMAPE": np.mean(2 * np.abs(err) / (np.abs(te.values) + np.abs(p))) * 100,
                     "MASE": np.mean(np.abs(err)) / scale})
        fan_store.setdefault(name, []).append(err)
    print(f"origin {te.index[0].date()} done")
bt = P.table(pd.DataFrame(recs), "backtest_all_origins")
board = bt.groupby("model")[["MAE", "RMSE", "sMAPE", "MASE"]].mean().sort_values("MASE")
board["MASE_std"] = bt.groupby("model")["MASE"].std()
board.reset_index().to_csv(P.out / "backtest_leaderboard.csv", index=False)
print(board.round(3))
P.results["leaderboard"] = board.round(3).reset_index().to_dict(orient="records")
best = board.index[0]
P.results["best_model"] = best

fig, axes = plt.subplots(1, 2, figsize=(17, 5))
board["MASE"].sort_values(ascending=False).plot.barh(ax=axes[0], xerr=board["MASE_std"], color="#55a868")
axes[0].axvline(1, c="k", ls="--", lw=1)
axes[0].set(title="Mean MASE over 12 origins (<1 beats seasonal naive)", xlabel="MASE")
for name, g in bt.groupby("model"):
    axes[1].plot(pd.to_datetime(g["origin"]), g["MAE"], "o-", label=name, lw=1)
axes[1].set(title="MAE at each forecast origin", ylabel="MAE (rentals/day)")
axes[1].legend(fontsize=7)
P.save(fig, "04_backtest")

# %% [markdown]
# ## 6 · Deployment — final 14-day forecast with an empirical prediction fan
# Interval width per horizon step comes from the 12 backtest errors at that step: an empirical 80 % band (10th–90th
# percentile) and the full min–max range. With only 12 samples these are rough, but they reflect real out-of-sample error.

# %%
P.section("6 · Final forecast")
errs = np.vstack(fan_store[best])  # origins × horizon
q = {p: np.quantile(errs, p, axis=0) for p in (0.0, 0.1, 0.9, 1.0)}  # only 12 errors per step -> 80 % band + full range
fut_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
fc = MODELS[best](y, H)
fcast = pd.DataFrame({"forecast": fc, "lo_range": fc + q[0.0], "lo80": fc + q[0.1], "hi80": fc + q[0.9],
                      "hi_range": fc + q[1.0]}, index=fut_idx).round(0)
fcast.to_csv(P.out / "final_forecast.csv")
print(fcast)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y.iloc[-120:], c="k", lw=1.2, label="history")
ax.fill_between(fut_idx, fcast["lo_range"], fcast["hi_range"], color="#4c72b0", alpha=.15, label="range of backtest errors")
ax.fill_between(fut_idx, fcast["lo80"], fcast["hi80"], color="#4c72b0", alpha=.3, label="empirical 80 % band")
ax.plot(fut_idx, fcast["forecast"], c="#c44e52", lw=2, label=f"forecast — {best}")
ax.set(title="14-day forecast fan (Jan 2013)", ylabel="rentals/day")
ax.legend()
P.save(fig, "05_forecast_fan")

R = P.results
naive_mase = board.loc["Seasonal naive (t-7)", "MASE"] if "Seasonal naive (t-7)" in board.index else 1.0
P.finish(f"""# 12 · Time-Series Forecasting — Results

* {R['n_days']} days of Capital Bikeshare rentals (2011-01-01 → 2012-12-31), horizon **{H} days**, 12 rolling origins.
* Stationarity: ADF p={R['stationarity']['adf_p_level']:.3f} (level), KPSS p={R['stationarity']['kpss_p_level']:.3f}, ADF p={R['stationarity']['adf_p_first_diff']:.2e} after differencing → d = 1.

## Backtest leaderboard (mean over origins)
{board.round(3).to_markdown()}

## Final forecast (first days of 2013)
{fcast.head(7).to_markdown()}

**Paraphrase:** the best approach (**{best}**) has MASE {board.loc[best, 'MASE']:.2f} (the seasonal-naive rule scores {naive_mase:.2f}),
so it is {(1 - board.loc[best, 'MASE'] / naive_mase) * 100:.0f}% more accurate than "same day last week". Weather is deliberately left
out (it isn't known 14 days ahead), and the EDA shows it drives large day-to-day swings, so it is a likely source of the biggest misses. That is why the forecast fan is wide, and why
adding a weather forecast feed is the most valuable next step.
""")
