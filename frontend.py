import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ================= CONFIG =================
API_URL = "https://fraud-detection-api-mtyt.onrender.com/predict"

st.set_page_config(page_title="FraudShield", layout="wide")

# ================= SESSION STATE =================
if "input_data" not in st.session_state:
    st.session_state["input_data"] = {f"V{i}": 0.0 for i in range(1, 29)}
    st.session_state["input_data"]["Amount"] = 0.0
    st.session_state["input_data"]["Time"] = 0.0

input_data = st.session_state["input_data"]

# ================= UI =================
st.title("💳 FraudShield - AI Fraud Detection")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Transaction Simulator")

    # ✅ REQUIRED FIELDS
    input_data["Time"] = st.number_input("Transaction Time", value=float(input_data.get("Time", 0.0)))
    input_data["Amount"] = st.number_input("Transaction Amount ($)", value=float(input_data.get("Amount", 120.0)))

    merchant = st.text_input("Merchant", "Amazon")
    location = st.text_input("Location", "New York")

    # ===== Buttons =====
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🔥 Simulate Fraud"):
            fraud_data = {f"V{i}": 0.0 for i in range(1, 29)}
            fraud_data["V14"] = -6.5
            fraud_data["Amount"] = 300.0
            fraud_data["Time"] = 100000.0  # ✅ IMPORTANT
            st.session_state["input_data"] = fraud_data
            st.success("Fraud scenario loaded")

    with c2:
        if st.button("✅ Simulate Legit"):
            legit_data = {f"V{i}": 0.0 for i in range(1, 29)}
            legit_data["Amount"] = 50.0
            legit_data["Time"] = 50000.0  # ✅ IMPORTANT
            st.session_state["input_data"] = legit_data
            st.success("Legit scenario loaded")

    st.markdown("---")

    analyze = st.button("🚀 Analyze Transaction")

with col2:
    st.subheader("💳 Card Preview")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1F2937, #111827);
                padding:20px; border-radius:16px; color:white'>
        <h4>**** **** **** 1234</h4>
        <p>Card Holder</p>
        <p>VALID THRU 12/28</p>
    </div>
    """, unsafe_allow_html=True)

# ================= ANALYSIS =================
if analyze:

    # ✅ FIXED PAYLOAD (INCLUDING TIME)
    payload = {f"V{i}": float(input_data.get(f"V{i}", 0.0)) for i in range(1, 29)}
    payload["Amount"] = float(input_data.get("Amount", 0.0))
    payload["Time"] = float(input_data.get("Time", 0.0))  # 🔥 FIX

    st.write("📦 Payload:", payload)

    try:
        response = requests.post(API_URL, json=payload, timeout=30)

        st.write("🔢 Status Code:", response.status_code)
        st.write("📩 Raw Response:", response.text)

        if response.status_code == 200:
            result = response.json()

            fraud_prob = result.get("fraud_probability", 0)
            label = result.get("predicted_label", "unknown")
            risk = result.get("risk_level", "unknown")

            # ================= RESULT DISPLAY =================
            st.markdown("## 🔍 Prediction Result")

            colA, colB = st.columns(2)

            with colA:
                st.metric("Fraud Probability", f"{fraud_prob:.2%}")
                st.metric("Prediction", label.upper())
                st.metric("Risk Level", risk.upper())

            # ================= GAUGE =================
            with colB:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_prob * 100,
                    title={'text': "Fraud Risk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 40], 'color': "green"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"},
                        ]
                    }
                ))

                st.plotly_chart(fig, use_container_width=True)

            # ================= SHAP =================
            shap_data = result.get("shap_top_features", [])

            if shap_data:
                st.markdown("## 📊 Top SHAP Features")

                df = pd.DataFrame(shap_data)

                fig = go.Figure(go.Bar(
                    x=df["shap_value"],
                    y=df["feature"],
                    orientation='h'
                ))

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ API Error — check payload or backend logs")

    except Exception as e:
        st.error(f"🚨 Connection Error: {e}")