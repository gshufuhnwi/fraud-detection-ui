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
    st.session_state["input_data"]["Amount"] = 120.0
    st.session_state["input_data"]["Time"] = 0.0

# Initialize UI keys if not present
if "amount_input" not in st.session_state:
    st.session_state["amount_input"] = st.session_state["input_data"]["Amount"]

if "time_input" not in st.session_state:
    st.session_state["time_input"] = st.session_state["input_data"]["Time"]

input_data = st.session_state["input_data"]

# ================= HEADER =================
st.title("💳 FraudShield - AI Fraud Detection")

mode = st.radio("Mode", ["Simple", "Advanced"])

col1, col2 = st.columns([3, 1])

# ================= INPUT =================
with col1:
    st.subheader("Transaction Simulator")

    # ===== FIXED INPUTS =====
    amount = st.number_input(
        "Transaction Amount ($)",
        value=float(st.session_state["amount_input"]),
        key="amount_input"
    )

    time = st.number_input(
        "Transaction Time",
        value=float(st.session_state["time_input"]),
        key="time_input"
    )

    # Sync UI → session_state
    input_data["Amount"] = amount
    input_data["Time"] = time

    # ===== ADVANCED FEATURES =====
    if mode == "Advanced":
        with st.expander("⚙️ Advanced Feature Input (V1–V28)"):
            cols = st.columns(4)
            for i in range(1, 29):
                key = f"v{i}_input"

                if key not in st.session_state:
                    st.session_state[key] = input_data.get(f"V{i}", 0.0)

                with cols[(i - 1) % 4]:
                    val = st.number_input(f"V{i}", key=key)

                input_data[f"V{i}"] = val

    # ===== BUTTONS =====
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🔥 Simulate Fraud"):
            st.session_state["amount_input"] = 300.0
            st.session_state["time_input"] = 100000.0

            for i in range(1, 29):
                st.session_state[f"v{i}_input"] = 0.0
            st.session_state["v14_input"] = -6.5

            st.success("Fraud scenario loaded")

    with c2:
        if st.button("✅ Simulate Legit"):
            st.session_state["amount_input"] = 50.0
            st.session_state["time_input"] = 50000.0

            for i in range(1, 29):
                st.session_state[f"v{i}_input"] = 0.0

            st.success("Legit scenario loaded")

    analyze = st.button("🚀 Analyze Transaction")

# ================= CARD UI =================
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

# ================= PREDICTION =================
if analyze:

    payload = {f"V{i}": float(input_data.get(f"V{i}", 0.0)) for i in range(1, 29)}
    payload["Amount"] = float(input_data["Amount"])
    payload["Time"] = float(input_data["Time"])

    try:
        response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            fraud_prob = result.get("fraud_probability", 0)
            label = result.get("predicted_label", "unknown")
            risk = result.get("risk_level", "unknown")

            st.markdown("## 🔍 Prediction Result")

            colA, colB = st.columns(2)

            with colA:
                st.metric("Fraud Probability", f"{fraud_prob:.2%}")
                st.metric("Prediction", label.upper())
                st.metric("Risk Level", risk.upper())

            # ===== GAUGE =====
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

            # ===== SHAP =====
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
                st.info("No SHAP data available")

        else:
            st.error("❌ API Error — check backend logs")

    except Exception as e:
        st.error(f"🚨 Connection Error: {e}")

# ================= CSV BATCH =================
st.markdown("---")
st.markdown("## 📂 Batch Fraud Detection (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview:")
    st.dataframe(df.head())

    required_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.error(f"❌ Missing columns: {missing}")
    else:
        if st.button("🚀 Run Batch Prediction"):

            results = []
            progress = st.progress(0)

            for i, row in df.iterrows():
                try:
                    payload = row.to_dict()
                    response = requests.post(API_URL, json=payload)

                    if response.status_code == 200:
                        r = response.json()
                        results.append({
                            "fraud_probability": r["fraud_probability"],
                            "prediction": r["predicted_label"],
                            "risk_level": r["risk_level"]
                        })
                    else:
                        results.append({"fraud_probability": None, "prediction": "error", "risk_level": "error"})

                except:
                    results.append({"fraud_probability": None, "prediction": "failed", "risk_level": "failed"})

                progress.progress((i + 1) / len(df))

            results_df = pd.DataFrame(results)
            final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

            st.success("✅ Batch scoring complete!")
            st.dataframe(final_df)

            csv = final_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Results",
                csv,
                "fraud_predictions.csv",
                "text/csv"
            )