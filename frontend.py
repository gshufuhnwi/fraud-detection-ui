import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ================= CONFIG =================
API_URL = "https://fraud-detection-api-mtyt.onrender.com/predict"

st.set_page_config(page_title="FraudShield", layout="wide")

# ================= SESSION STATE =================
if "amount_input" not in st.session_state:
    st.session_state["amount_input"] = 120.0

if "time_input" not in st.session_state:
    st.session_state["time_input"] = 0.0

for i in range(1, 29):
    if f"v{i}_input" not in st.session_state:
        st.session_state[f"v{i}_input"] = 0.0

# ================= CALLBACKS =================
def load_fraud():
    st.session_state["amount_input"] = 300.0
    st.session_state["time_input"] = 100000.0
    for i in range(1, 29):
        st.session_state[f"v{i}_input"] = 0.0
    st.session_state["v14_input"] = -6.5


def load_legit():
    st.session_state["amount_input"] = 50.0
    st.session_state["time_input"] = 50000.0
    for i in range(1, 29):
        st.session_state[f"v{i}_input"] = 0.0

# ================= HEADER =================
st.title("💳 FraudShield - AI Fraud Detection Platform")

tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Dashboard"])

# =====================================================
# 🔍 TAB 1: PREDICTION
# =====================================================
with tab1:

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Transaction Simulator")

        amount = st.number_input("Transaction Amount ($)", key="amount_input")
        time = st.number_input("Transaction Time", key="time_input")

        # Collect features
        input_data = {
            f"V{i}": st.session_state[f"v{i}_input"] for i in range(1, 29)
        }
        input_data["Amount"] = amount
        input_data["Time"] = time

        # Buttons
        c1, c2 = st.columns(2)
        with c1:
            st.button("🔥 Simulate Fraud", on_click=load_fraud)
        with c2:
            st.button("✅ Simulate Legit", on_click=load_legit)

        analyze = st.button("🚀 Analyze Transaction")

    # ===== CARD =====
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

    # ===== PREDICTION =====
    if analyze:
        try:
            response = requests.post(API_URL, json=input_data, timeout=60)

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

                # Gauge
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

                # SHAP
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
                st.error("❌ API Error")

        except Exception as e:
            st.error(f"🚨 {e}")

    # ===== CSV =====
    st.markdown("---")
    st.markdown("## 📂 Batch Fraud Detection (CSV Upload)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        required_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        if not all(col in df.columns for col in required_cols):
            st.error("❌ Missing required columns")
        else:
            if st.button("🚀 Run Batch Prediction"):
                results = []

                for _, row in df.iterrows():
                    try:
                        r = requests.post(API_URL, json=row.to_dict()).json()
                        results.append({
                            "fraud_probability": r["fraud_probability"],
                            "prediction": r["predicted_label"],
                            "risk_level": r["risk_level"]
                        })
                    except:
                        results.append({"prediction": "error"})

                results_df = pd.DataFrame(results)
                final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

                st.dataframe(final_df)

                st.download_button(
                    "📥 Download Results",
                    final_df.to_csv(index=False),
                    "fraud_results.csv"
                )

# =====================================================
# 📊 TAB 2: DASHBOARD
# =====================================================
with tab2:

    st.subheader("📊 Fraud Analytics Dashboard")

    uploaded_file = st.file_uploader("Upload Scored CSV", type=["csv"], key="dashboard")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "fraud_probability" not in df.columns:
            st.warning("Upload a CSV with predictions first.")
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("Avg Fraud Prob", f"{df['fraud_probability'].mean():.2%}")
            col2.metric("Max Fraud", f"{df['fraud_probability'].max():.2%}")
            col3.metric("Transactions", len(df))

            st.markdown("### Fraud Probability Distribution")
            fig = px.histogram(df, x="fraud_probability", nbins=30)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Risk Breakdown")
            if "risk_level" in df.columns:
                fig = px.pie(df, names="risk_level")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Top Risk Transactions")
            st.dataframe(df.sort_values("fraud_probability", ascending=False).head(10))