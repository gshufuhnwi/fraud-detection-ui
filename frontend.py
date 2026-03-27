import streamlit as st
import requests
import pandas as pd

API_URL = "https://fraud-detection-api-mtyt.onrender.com/predict"

st.set_page_config(page_title="FraudShield", layout="wide")

# ------------------ STYLE ------------------
st.markdown("""
<style>
body {
    background-color: #0B0F19;
}
.block-container {
    padding-top: 1rem;
}
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.title {
    font-size: 32px;
    font-weight: bold;
    color: white;
}
.subtitle {
    color: #9CA3AF;
}
.section {
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
col1, col2 = st.columns([1, 5])

with col1:
    st.markdown("💳")

with col2:
    st.markdown('<div class="title">FraudShield</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time transaction risk analysis</div>', unsafe_allow_html=True)

st.divider()

# ------------------ INPUT FORM ------------------
st.subheader("Transaction Input")

cols = st.columns(4)
input_data = {}

for i in range(1, 29):
    with cols[i % 4]:
        input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"V{i}")

input_data["Amount"] = st.number_input("Amount", value=0.0)

# ------------------ BUTTON ------------------
if st.button("Analyze Transaction"):
    with st.spinner("Processing..."):

        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()

            prob = result["fraud_probability"]
            label = result["predicted_label"]
            risk = result["risk_level"]

            st.divider()

            # ------------------ SUMMARY CARDS ------------------
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric("Fraud Probability", f"{prob:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric("Prediction", label.upper())
                st.markdown('</div>', unsafe_allow_html=True)

            with c3:
                st.markdown('<div class="card">', unsafe_allow_html=True)

                if risk == "high":
                    st.error("HIGH RISK 🚨")
                elif risk == "medium":
                    st.warning("MEDIUM RISK ⚠️")
                else:
                    st.success("LOW RISK ✅")

                st.markdown('</div>', unsafe_allow_html=True)

            # ------------------ SHAP CHART ------------------
            st.markdown("### Feature Impact (SHAP)")

            shap_df = pd.DataFrame(result["shap_top_features"])
            shap_df["abs_val"] = shap_df["shap_value"].abs()
            shap_df = shap_df.sort_values("abs_val")

            st.bar_chart(
                shap_df.set_index("feature")["shap_value"]
            )

            # ------------------ DETAILS TABLE ------------------
            st.markdown("### Transaction Breakdown")

            st.dataframe(
                shap_df[["feature", "value", "shap_value"]],
                use_container_width=True
            )

        else:
            st.error("API error — check backend logs")