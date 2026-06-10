import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Explorer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Customer Explorer")

df = pd.read_csv("data/features.csv")

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

customer_id = st.selectbox(
    "Select Customer ID",
    df["Customer ID"].unique()
)

customer = df[df["Customer ID"] == customer_id]
st.dataframe(customer)

feature_cols = [
    "Recency", "Frequency", "Monetary",
    "AvgOrderValue", "ProductDiversity",
    "TotalQuantity", "CustomerLifetime",
    "AvgPurchaseInterval",
    "SingleOrderCustomer", "IsUKCustomer"
]

X = customer[feature_cols]
X_scaled = scaler.transform(X)

churn_prob = model.predict_proba(X_scaled)[0][1]

st.metric("Churn Probability", f"{churn_prob:.2%}")

if churn_prob >= 0.7:
    risk = "High Risk"
elif churn_prob >= 0.4:
    risk = "Medium Risk"
else:
    risk = "Low Risk"

st.subheader(f"Risk Segment: {risk}")

st.markdown("### Customer Profile Summary")

st.write(f"""
- Recency: {customer['Recency'].values[0]}
- Frequency: {customer['Frequency'].values[0]}
- Monetary: {customer['Monetary'].values[0]}
- Lifetime: {customer['CustomerLifetime'].values[0]}
""")

st.markdown("### Recommended Action")

if risk == "High Risk":
    st.error("📢 Send discount + reactivation campaign")
elif risk == "Medium Risk":
    st.warning("📧 Send engagement emails + offers")
else:
    st.success("🎯 Maintain loyalty program benefits")

