"""Streamlit what-if app for the churn model (CRISP-DM Phase 6: Deployment).

Run after training:   streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import engineer

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
st.set_page_config(page_title="Telco Churn Risk", page_icon="📉", layout="wide")

model_path = OUT / "churn_model.joblib"
if not model_path.exists():
    st.error("Model not found. Run `python churn_crispdm.py` first.")
    st.stop()
bundle = joblib.load(model_path)
model, threshold = bundle["model"], bundle["threshold"]
metrics = json.loads((OUT / "metrics.json").read_text()) if (OUT / "metrics.json").exists() else {}

st.title("📉 Telco Customer Churn — Risk Scorer")
st.caption(f"Model: **{bundle['model_name']}** · cost-optimal threshold **{threshold:.2f}** · "
           f"test ROC-AUC **{metrics.get('test_at_cost_threshold', {}).get('roc_auc', float('nan')):.3f}**")

tab_score, tab_eval, tab_data = st.tabs(["🔮 Score a customer", "📊 Model evaluation", "🧭 EDA & segments"])

with tab_score:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 5)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 85.0, 0.5)
        total = st.number_input("Total charges ($)", 0.0, 9000.0, float(round(tenure * monthly, 2)))
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
    with c2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        opts = ["No", "Yes", "No internet service"]
        sec = st.selectbox("Online security", opts)
        backup = st.selectbox("Online backup", opts)
        device = st.selectbox("Device protection", opts)
        tech = st.selectbox("Tech support", opts)
        tv = st.selectbox("Streaming TV", opts)
        movies = st.selectbox("Streaming movies", opts)
    with c3:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    row = pd.DataFrame([{
        "customerID": "what-if", "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi, "InternetService": internet, "OnlineSecurity": sec, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total}])
    p = float(model.predict_proba(engineer(row))[:, 1][0])
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{p:.1%}")
    m2.metric("Decision", "🚨 Send retention offer" if p >= threshold else "✅ No action")
    m3.metric("Threshold", f"{threshold:.2f}")
    st.progress(min(p, 1.0))
    if p >= threshold:
        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discounted 1- or 2-year contract.")
        if internet == "Fiber optic" and tech != "Yes":
            tips.append("Bundle free Tech Support / Online Security for 6 months.")
        if payment == "Electronic check":
            tips.append("Incentivise switching to automatic payment.")
        if tenure <= 6:
            tips.append("Enrol in a first-90-days onboarding programme.")
        st.info("**Suggested actions:**\n\n" + "\n".join(f"- {t}" for t in tips or ["Personal outreach call."]))

with tab_eval:
    fig_dir = OUT / "figures"
    for name in ["11_evaluation_dashboard", "09_cv_leaderboard", "12_permutation_importance",
                 "14_shap_beeswarm", "13_logreg_odds_ratios"]:
        f = fig_dir / f"{name}.png"
        if f.exists():
            st.image(str(f))
    lb = OUT / "cv_leaderboard.csv"
    if lb.exists():
        st.dataframe(pd.read_csv(lb).round(4))

with tab_data:
    for name in ["03_categorical_churn_rates", "02_numeric_distributions", "07_kmeans_segments",
                 "08_mutual_information", "06_outlier_boxplots"]:
        f = OUT / "figures" / f"{name}.png"
        if f.exists():
            st.image(str(f))
