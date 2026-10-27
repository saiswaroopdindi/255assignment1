"""Shared helpers for the Part 2 experiments (downloads, figure saving, result logging)."""
from __future__ import annotations

import json
import shutil
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PART2 = Path(__file__).resolve().parent
DATA = PART2 / "data"  # shared download cache for all projects (git-ignored)
DATA.mkdir(exist_ok=True)


def _ctx():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download(url: str, name: str | None = None) -> Path:
    """Download once into part2_ds_replication/data and return the local path."""
    dest = DATA / (name or url.rsplit("/", 1)[-1].replace("+", "_"))
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    print(f"Downloading {url}\n  -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (cmpe255-assignment)"})
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(req, context=_ctx(), timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:
        raise SystemExit(f"Download failed ({e}). Download {url} manually and save it as {dest}") from e
    tmp.replace(dest)
    return dest


def load_online_retail():
    """UCI Online Retail (same data as Kaggle 'E-Commerce Data' / 'Online Retail'), cached as parquet."""
    import zipfile

    import pandas as pd

    cache = DATA / "online_retail.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    z = download("https://archive.ics.uci.edu/static/public/352/online+retail.zip", "online_retail.zip")
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        zf.extract(name, DATA)
    print("Reading Excel file (one-time, ~1-2 min)...")
    df = pd.read_excel(DATA / name, dtype={"InvoiceNo": str, "StockCode": str, "Description": str})
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    df.to_parquet(cache, index=False)
    return df


class Project:
    """Creates outputs/ + outputs/figures for a project and collects results."""

    def __init__(self, base: Path, title: str):
        self.base = Path(base)
        self.out = self.base / "outputs"
        self.fig_dir = self.out / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict = {"project": title}
        self.t0 = time.time()
        print("=" * 90 + f"\n{title}\n" + "=" * 90)

    def section(self, title: str):
        print("\n" + "-" * 90 + f"\n{title}\n" + "-" * 90)

    def save(self, fig, name: str):
        path = self.fig_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        if IN_NOTEBOOK:
            plt.show()
        plt.close(fig)
        print(f"  [figure] outputs/figures/{name}.png")

    def table(self, df, name: str):
        df.to_csv(self.out / f"{name}.csv", index=False)
        return df

    def finish(self, report_md: str):
        self.results["runtime_seconds"] = round(time.time() - self.t0, 1)
        (self.out / "metrics.json").write_text(json.dumps(self.results, indent=2, default=str))
        (self.out / "RESULTS.md").write_text(report_md)
        print(f"\nWrote outputs/metrics.json and outputs/RESULTS.md ({self.results['runtime_seconds']}s)")


def base_dir(globals_dict) -> Path:
    """Folder of the running script, or the notebook's working directory."""
    f = globals_dict.get("__file__")
    return Path(f).resolve().parent if f else Path.cwd()


def add_part2_to_path():
    if str(PART2) not in sys.path:
        sys.path.insert(0, str(PART2))
