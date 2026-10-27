#!/usr/bin/env bash
# One-shot runner: creates a virtual env, installs deps, runs every experiment, builds notebooks.
# Usage (macOS/Linux):  cd assignment1 && bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# optional deep-learning section of Part 1 (comment out to skip, ~200 MB)
python -m pip install torch || echo "torch install skipped"

python tools/py_to_ipynb.py \
  part1_telco_churn/churn_crispdm.py \
  part2_ds_replication/01_nyc_taxi/nyc_taxi_trip_duration.py \
  part2_ds_replication/03_customer_segmentation/customer_segmentation.py \
  part2_ds_replication/04_market_basket/market_basket_mining.py \
  part2_ds_replication/06_anomaly_detection/anomaly_detection.py \
  part2_ds_replication/12_timeseries_forecasting/timeseries_forecasting.py

run () {
  echo; echo "################ $1 ################"
  ( cd "$(dirname "$1")" && python -u "$(basename "$1")" 2>&1 | tee run_log.txt ) || echo "!! $1 failed - see run_log.txt"
}
run part1_telco_churn/churn_crispdm.py                                  # ~5-10 min
run part2_ds_replication/12_timeseries_forecasting/timeseries_forecasting.py   # ~2 min
run part2_ds_replication/03_customer_segmentation/customer_segmentation.py     # ~3 min (first run reads Excel)
run part2_ds_replication/04_market_basket/market_basket_mining.py             # ~2 min
run part2_ds_replication/06_anomaly_detection/anomaly_detection.py            # ~5 min (downloads 150 MB)
run part2_ds_replication/01_nyc_taxi/nyc_taxi_trip_duration.py                # ~8 min (downloads 50 MB)

echo; echo "All done. Dashboards:"
echo "  streamlit run part1_telco_churn/app.py"
echo "  streamlit run part2_ds_replication/dashboard.py"
