import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "https://fraud-detection-api-mtyt.onrender.com/predict"

st.set_page_config(page_title="FraudShield", layout="wide")

# ------------------ STYLE ------------------
st.markdown("""
<style>
body {
    background-color: #0B0F19;
}
.card {
    background: linear-gradient(135deg, #1F2937, #111827);
    padding: 20px;
    border-radius: 16px;
    color: white;
}
.title {
    font-size: 32px;
    font-weight: bold;
}
.subtitle {
    color: #9CA3AF;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="title">💳 FraudShield</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered fraud detection dashboard</div>', unsafe_allow_html=True)

st.divider()

# ------------------ CREDIT CARD UI ------------------
st.subheader("💳 Transaction Simulator")

col1, col2 = st.columns([2, 1])

with col1:
    amount = st.number_input("Transaction Amount ($)", value=120.0)

    colA, colB = st.columns(2)
    with colA:
        merchant = st.text_input("Merchant", "Amazon")
    with colB:
        location = st.text_input("Location", "New York")

with col2:
    st.markdown("""
    <div class="card">
        <h4>💳 Card Preview</h4>
        <p>**** **** **** 1234</p>
        <p>Card Holder</p>
        <p>VALID THRU 12/28</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------ DEFAULT FEATURES ------------------
input_data = {f"V{i}": 0.0 for i in range(1, 29)}
input_data["Amount"] = amount

# ------------------ DEMO BUTTONS ------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🔥 Simulate Fraud"):
        input_data.update({
            "V14": -6.5,
            "V10": -6.2,
            "V12": -8.2,
            "Amount": 300
        })

with col2:
    if st.button("✅ Simulate Legit"):
        input_data.update({
            "Amount": 50
        })

# ------------------ ADVANCED FEATURES ------------------
with st.expander("⚙️ Advanced Features (Optional)"):
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[i % 4]:
            input_data[f"V{i}"] = st.number_input(f"V{i}", value=input_data[f"V{i}"], key=f"V{i}")

# ------------------ GAUGE FUNCTION ------------------
def create_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Fraud Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"},
            ],
        }
    ))
    return fig

# ------------------ PREDICT ------------------
if st.button("🚀 Analyze Transaction"):
    with st.spinner("Analyzing..."):

        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()

            prob = result["fraud_probability"]
            label = result["predicted_label"]
            risk = result["risk_level"]

            st.divider()

            # ------------------ SUMMARY ------------------
            col1, col2 = st.columns([1, 2])

            with col1:
                st.plotly_chart(create_gauge(prob), use_container_width=True)

            with col2:
                st.markdown("### 💳 Transaction Summary")
                st.write(f"**Amount:** ${input_data['Amount']}")
                st.write(f"**Prediction:** {label.upper()}")
                st.write(f"**Probability:** {prob:.4f}")

                if risk == "high":
                    st.error("🚨 HIGH RISK")
                elif risk == "medium":
                    st.warning("⚠️ MEDIUM RISK")
                else:
                    st.success("✅ LOW RISK")

            # ------------------ SHAP ------------------
            st.subheader("📊 Key Risk Drivers")

            shap_df = pd.DataFrame(result["shap_top_features"])
            shap_df["abs_val"] = shap_df["shap_value"].abs()
            shap_df = shap_df.sort_values("abs_val")

            st.bar_chart(
                shap_df.set_index("feature")["shap_value"]
            )

            st.dataframe(
                shap_df[["feature", "value", "shap_value"]],
                use_container_width=True
            )

        else:
            st.error("❌ API error — check backend logs")